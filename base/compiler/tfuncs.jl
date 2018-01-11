# This file is a part of Julia. License is MIT: https://julialang.org/license

##################
# InferenceState #
##################

const LineNum = Int

mutable struct InferenceState
    params::Params # describes how to compute the result
    result::InferenceResult # remember where to put the result
    linfo::MethodInstance # used here for the tuple (specTypes, env, Method) and world-age validity
    sp::SimpleVector     # static parameters
    mod::Module
    currpc::LineNum

    # info on the state of inference and the linfo
    src::CodeInfo
    min_valid::UInt
    max_valid::UInt
    nargs::Int
    stmt_types::Vector{Any}
    stmt_edges::Vector{Any}
    # return type
    bestguess #::Type
    # current active instruction pointers
    ip::BitSet
    pc´´::LineNum
    nstmts::Int
    # current exception handler info
    cur_hand #::Tuple{LineNum, Tuple{LineNum, ...}}
    handler_at::Vector{Any}
    n_handlers::Int
    # ssavalue sparsity and restart info
    ssavalue_uses::Vector{BitSet}
    ssavalue_defs::Vector{LineNum}
    vararg_type_container #::Type

    backedges::Vector{Tuple{InferenceState, LineNum}} # call-graph backedges connecting from callee to caller
    callers_in_cycle::Vector{InferenceState}
    parent::Union{Nothing, InferenceState}

    const_api::Bool
    const_ret::Bool

    # TODO: move these to InferenceResult / Params?
    optimize::Bool
    cached::Bool
    limited::Bool
    inferred::Bool
    dont_work_on_me::Bool

    # src is assumed to be a newly-allocated CodeInfo, that can be modified in-place to contain intermediate results
    function InferenceState(result::InferenceResult, src::CodeInfo,
                            optimize::Bool, cached::Bool, params::Params)
        linfo = result.linfo
        code = src.code::Array{Any,1}
        toplevel = !isa(linfo.def, Method)

        if !toplevel && isempty(linfo.sparam_vals) && !isempty(linfo.def.sparam_syms)
            # linfo is unspecialized
            sp = Any[]
            sig = linfo.def.sig
            while isa(sig, UnionAll)
                push!(sp, sig.var)
                sig = sig.body
            end
            sp = svec(sp...)
        else
            sp = linfo.sparam_vals
        end

        nssavalues = src.ssavaluetypes::Int
        src.ssavaluetypes = Any[ NOT_FOUND for i = 1:nssavalues ]

        n = length(code)
        s_edges = Any[ () for i = 1:n ]
        s_types = Any[ () for i = 1:n ]

        # initial types
        nslots = length(src.slotnames)
        argtypes = get_argtypes(result)
        vararg_type_container = nothing
        nargs = length(argtypes)
        s_argtypes = VarTable(uninitialized, nslots)
        src.slottypes = Vector{Any}(uninitialized, nslots)
        for i in 1:nslots
            at = (i > nargs) ? Bottom : argtypes[i]
            if !toplevel && linfo.def.isva && i == nargs
                if !(at == Tuple) # would just be a no-op
                    vararg_type_container = limit_tuple_depth(params, unwrap_unionall(at)) # TODO: should be limiting tuple depth much earlier than here
                    vararg_type = tuple_tfunc(vararg_type_container) # returns a Const object, if applicable
                    at = rewrap(vararg_type, linfo.specTypes)
                end
            end
            s_argtypes[i] = VarState(at, i > nargs)
            src.slottypes[i] = at
        end
        s_types[1] = s_argtypes

        ssavalue_uses = find_ssavalue_uses(code, nssavalues)
        ssavalue_defs = find_ssavalue_defs(code, nssavalues)

        # exception handlers
        cur_hand = ()
        handler_at = Any[ () for i=1:n ]
        n_handlers = 0

        W = BitSet()
        push!(W, 1) #initial pc to visit

        if !toplevel
            meth = linfo.def
            inmodule = meth.module
        else
            inmodule = linfo.def::Module
        end

        if cached && !toplevel
            min_valid = min_world(linfo.def)
            max_valid = max_world(linfo.def)
        else
            min_valid = typemax(UInt)
            max_valid = typemin(UInt)
        end
        frame = new(
            params, result, linfo,
            sp, inmodule, 0,
            src, min_valid, max_valid,
            nargs, s_types, s_edges,
            Union{}, W, 1, n,
            cur_hand, handler_at, n_handlers,
            ssavalue_uses, ssavalue_defs, vararg_type_container,
            Vector{Tuple{InferenceState,LineNum}}(), # backedges
            Vector{InferenceState}(), # callers_in_cycle
            #=parent=#nothing,
            false, false, optimize, cached, false, false, false)
        result.result = frame
        cached && push!(params.cache, result)
        return frame
    end
end

function InferenceState(linfo::MethodInstance, optimize::Bool, cached::Bool, params::Params)
    return InferenceState(InferenceResult(linfo), optimize, cached, params)
end

function InferenceState(result::InferenceResult, optimize::Bool, cached::Bool, params::Params)
    # prepare an InferenceState object for inferring lambda
    src = retrieve_code_info(result.linfo)
    src === nothing && return nothing
    validate_code_in_debug_mode(result.linfo, src, "lowered")
    return InferenceState(result, src, optimize, cached, params)
end

_topmod(sv::InferenceState) = _topmod(sv.mod)

# work towards converging the valid age range for sv
function update_valid_age!(min_valid::UInt, max_valid::UInt, sv::InferenceState)
    sv.min_valid = max(sv.min_valid, min_valid)
    sv.max_valid = min(sv.max_valid, max_valid)
    @assert(!isa(sv.linfo.def, Method) ||
            !sv.cached ||
            sv.min_valid <= sv.params.world <= sv.max_valid,
            "invalid age range update")
    nothing
end

update_valid_age!(edge::InferenceState, sv::InferenceState) = update_valid_age!(edge.min_valid, edge.max_valid, sv)
update_valid_age!(li::MethodInstance, sv::InferenceState) = update_valid_age!(min_world(li), max_world(li), sv)

function record_ssa_assign(ssa_id::Int, @nospecialize(new), frame::InferenceState)
    old = frame.src.ssavaluetypes[ssa_id]
    if old === NOT_FOUND || !(new ⊑ old)
        frame.src.ssavaluetypes[ssa_id] = tmerge(old, new)
        W = frame.ip
        s = frame.stmt_types
        for r in frame.ssavalue_uses[ssa_id]
            if s[r] !== () # s[r] === () => unreached statement
                if r < frame.pc´´
                    frame.pc´´ = r
                end
                push!(W, r)
            end
        end
    end
    nothing
end

function add_backedge!(frame::InferenceState, caller::InferenceState, currpc::Int)
    update_valid_age!(frame, caller)
    backedge = (caller, currpc)
    contains_is(frame.backedges, backedge) || push!(frame.backedges, backedge)
    return frame
end

# temporarily accumulate our edges to later add as backedges in the callee
function add_backedge!(li::MethodInstance, caller::InferenceState)
    isa(caller.linfo.def, Method) || return # don't add backedges to toplevel exprs
    if caller.stmt_edges[caller.currpc] === ()
        caller.stmt_edges[caller.currpc] = []
    end
    push!(caller.stmt_edges[caller.currpc], li)
    update_valid_age!(li, caller)
    nothing
end

# used to temporarily accumulate our no method errors to later add as backedges in the callee method table
function add_mt_backedge!(mt::MethodTable, @nospecialize(typ), caller::InferenceState)
    isa(caller.linfo.def, Method) || return # don't add backedges to toplevel exprs
    if caller.stmt_edges[caller.currpc] === ()
        caller.stmt_edges[caller.currpc] = []
    end
    push!(caller.stmt_edges[caller.currpc], mt)
    push!(caller.stmt_edges[caller.currpc], typ)
    nothing
end

function is_specializable_vararg_slot(@nospecialize(arg), sv::InferenceState)
    return (isa(arg, Slot) && slot_id(arg) == sv.nargs &&
            isa(sv.vararg_type_container, DataType))
end


function print_callstack(sv::InferenceState)
    while sv !== nothing
        print(sv.linfo)
        sv.limited && print("  [limited]")
        !sv.cached && print("  [uncached]")
        println()
        for cycle in sv.callers_in_cycle
            print(' ', cycle.linfo)
            cycle.limited && print("  [limited]")
            println()
        end
        sv = sv.parent
    end
end

#############
# constants #
#############

const AbstractEvalConstant = Const

const _NAMEDTUPLE_NAME = NamedTuple.body.body.name

const INT_INF = typemax(Int) # integer infinity

const N_IFUNC = reinterpret(Int32, arraylen) + 1
const T_IFUNC = Vector{Tuple{Int, Int, Any}}(uninitialized, N_IFUNC)
const T_IFUNC_COST = Vector{Int}(uninitialized, N_IFUNC)
const T_FFUNC_KEY = Vector{Any}()
const T_FFUNC_VAL = Vector{Tuple{Int, Int, Any}}()
const T_FFUNC_COST = Vector{Int}()

const DATATYPE_NAME_FIELDINDEX = fieldindex(DataType, :name)
const DATATYPE_PARAMETERS_FIELDINDEX = fieldindex(DataType, :parameters)
const DATATYPE_TYPES_FIELDINDEX = fieldindex(DataType, :types)
const DATATYPE_SUPER_FIELDINDEX = fieldindex(DataType, :super)
const DATATYPE_MUTABLE_FIELDINDEX = fieldindex(DataType, :mutable)

const TYPENAME_NAME_FIELDINDEX = fieldindex(TypeName, :name)
const TYPENAME_MODULE_FIELDINDEX = fieldindex(TypeName, :module)
const TYPENAME_WRAPPER_FIELDINDEX = fieldindex(TypeName, :wrapper)

##########
# tfuncs #
##########

function add_tfunc(f::IntrinsicFunction, minarg::Int, maxarg::Int, @nospecialize(tfunc), cost::Int)
    idx = reinterpret(Int32, f) + 1
    T_IFUNC[idx] = (minarg, maxarg, tfunc)
    T_IFUNC_COST[idx] = cost
end
# TODO: add @nospecialize on `f` and declare its type as `Builtin` when that's supported
function add_tfunc(f::Function, minarg::Int, maxarg::Int, @nospecialize(tfunc), cost::Int)
    push!(T_FFUNC_KEY, f)
    push!(T_FFUNC_VAL, (minarg, maxarg, tfunc))
    push!(T_FFUNC_COST, cost)
end

add_tfunc(throw, 1, 1, (@nospecialize(x)) -> Bottom, 0)

# the inverse of typeof_tfunc
# returns (type, isexact)
# if isexact is false, the actual runtime type may (will) be a subtype of t
function instanceof_tfunc(@nospecialize(t))
    if t === Bottom || t === typeof(Bottom)
        return Bottom, true
    elseif isa(t, Const)
        if isa(t.val, Type)
            return t.val, true
        end
    elseif isType(t)
        tp = t.parameters[1]
        return tp, !has_free_typevars(tp)
    elseif isa(t, UnionAll)
        t′ = unwrap_unionall(t)
        t′′, isexact = instanceof_tfunc(t′)
        return rewrap_unionall(t′′, t), isexact
    elseif isa(t, Union)
        ta, isexact_a = instanceof_tfunc(t.a)
        tb, isexact_b = instanceof_tfunc(t.b)
        return Union{ta, tb}, false # at runtime, will be exactly one of these
    end
    return Any, false
end
bitcast_tfunc(@nospecialize(t), @nospecialize(x)) = instanceof_tfunc(t)[1]
math_tfunc(@nospecialize(x)) = widenconst(x)
math_tfunc(@nospecialize(x), @nospecialize(y)) = widenconst(x)
math_tfunc(@nospecialize(x), @nospecialize(y), @nospecialize(z)) = widenconst(x)
fptoui_tfunc(@nospecialize(t), @nospecialize(x)) = bitcast_tfunc(t, x)
fptosi_tfunc(@nospecialize(t), @nospecialize(x)) = bitcast_tfunc(t, x)
function fptoui_tfunc(@nospecialize(x))
    T = widenconst(x)
    T === Float64 && return UInt64
    T === Float32 && return UInt32
    T === Float16 && return UInt16
    return Any
end
function fptosi_tfunc(@nospecialize(x))
    T = widenconst(x)
    T === Float64 && return Int64
    T === Float32 && return Int32
    T === Float16 && return Int16
    return Any
end

    ## conversion ##
add_tfunc(bitcast, 2, 2, bitcast_tfunc, 1)
add_tfunc(sext_int, 2, 2, bitcast_tfunc, 1)
add_tfunc(zext_int, 2, 2, bitcast_tfunc, 1)
add_tfunc(trunc_int, 2, 2, bitcast_tfunc, 1)
add_tfunc(fptoui, 1, 2, fptoui_tfunc, 1)
add_tfunc(fptosi, 1, 2, fptosi_tfunc, 1)
add_tfunc(uitofp, 2, 2, bitcast_tfunc, 1)
add_tfunc(sitofp, 2, 2, bitcast_tfunc, 1)
add_tfunc(fptrunc, 2, 2, bitcast_tfunc, 1)
add_tfunc(fpext, 2, 2, bitcast_tfunc, 1)
    ## arithmetic ##
add_tfunc(neg_int, 1, 1, math_tfunc, 1)
add_tfunc(add_int, 2, 2, math_tfunc, 1)
add_tfunc(sub_int, 2, 2, math_tfunc, 1)
add_tfunc(mul_int, 2, 2, math_tfunc, 4)
add_tfunc(sdiv_int, 2, 2, math_tfunc, 30)
add_tfunc(udiv_int, 2, 2, math_tfunc, 30)
add_tfunc(srem_int, 2, 2, math_tfunc, 30)
add_tfunc(urem_int, 2, 2, math_tfunc, 30)
add_tfunc(add_ptr, 2, 2, math_tfunc, 1)
add_tfunc(sub_ptr, 2, 2, math_tfunc, 1)
add_tfunc(neg_float, 1, 1, math_tfunc, 1)
add_tfunc(add_float, 2, 2, math_tfunc, 1)
add_tfunc(sub_float, 2, 2, math_tfunc, 1)
add_tfunc(mul_float, 2, 2, math_tfunc, 4)
add_tfunc(div_float, 2, 2, math_tfunc, 20)
add_tfunc(rem_float, 2, 2, math_tfunc, 20)
add_tfunc(fma_float, 3, 3, math_tfunc, 5)
add_tfunc(muladd_float, 3, 3, math_tfunc, 5)
    ## fast arithmetic ##
add_tfunc(neg_float_fast, 1, 1, math_tfunc, 1)
add_tfunc(add_float_fast, 2, 2, math_tfunc, 1)
add_tfunc(sub_float_fast, 2, 2, math_tfunc, 1)
add_tfunc(mul_float_fast, 2, 2, math_tfunc, 2)
add_tfunc(div_float_fast, 2, 2, math_tfunc, 10)
add_tfunc(rem_float_fast, 2, 2, math_tfunc, 10)
    ## bitwise operators ##
add_tfunc(and_int, 2, 2, math_tfunc, 1)
add_tfunc(or_int, 2, 2, math_tfunc, 1)
add_tfunc(xor_int, 2, 2, math_tfunc, 1)
add_tfunc(not_int, 1, 1, math_tfunc, 1)
add_tfunc(shl_int, 2, 2, math_tfunc, 1)
add_tfunc(lshr_int, 2, 2, math_tfunc, 1)
add_tfunc(ashr_int, 2, 2, math_tfunc, 1)
add_tfunc(bswap_int, 1, 1, math_tfunc, 1)
add_tfunc(ctpop_int, 1, 1, math_tfunc, 1)
add_tfunc(ctlz_int, 1, 1, math_tfunc, 1)
add_tfunc(cttz_int, 1, 1, math_tfunc, 1)
add_tfunc(checked_sdiv_int, 2, 2, math_tfunc, 40)
add_tfunc(checked_udiv_int, 2, 2, math_tfunc, 40)
add_tfunc(checked_srem_int, 2, 2, math_tfunc, 40)
add_tfunc(checked_urem_int, 2, 2, math_tfunc, 40)
    ## functions ##
add_tfunc(abs_float, 1, 1, math_tfunc, 2)
add_tfunc(copysign_float, 2, 2, math_tfunc, 2)
add_tfunc(flipsign_int, 2, 2, math_tfunc, 1)
add_tfunc(ceil_llvm, 1, 1, math_tfunc, 10)
add_tfunc(floor_llvm, 1, 1, math_tfunc, 10)
add_tfunc(trunc_llvm, 1, 1, math_tfunc, 10)
add_tfunc(rint_llvm, 1, 1, math_tfunc, 10)
add_tfunc(sqrt_llvm, 1, 1, math_tfunc, 20)
    ## same-type comparisons ##
cmp_tfunc(@nospecialize(x), @nospecialize(y)) = Bool
add_tfunc(eq_int, 2, 2, cmp_tfunc, 1)
add_tfunc(ne_int, 2, 2, cmp_tfunc, 1)
add_tfunc(slt_int, 2, 2, cmp_tfunc, 1)
add_tfunc(ult_int, 2, 2, cmp_tfunc, 1)
add_tfunc(sle_int, 2, 2, cmp_tfunc, 1)
add_tfunc(ule_int, 2, 2, cmp_tfunc, 1)
add_tfunc(eq_float, 2, 2, cmp_tfunc, 2)
add_tfunc(ne_float, 2, 2, cmp_tfunc, 2)
add_tfunc(lt_float, 2, 2, cmp_tfunc, 2)
add_tfunc(le_float, 2, 2, cmp_tfunc, 2)
add_tfunc(fpiseq, 2, 2, cmp_tfunc, 1)
add_tfunc(fpislt, 2, 2, cmp_tfunc, 1)
add_tfunc(eq_float_fast, 2, 2, cmp_tfunc, 1)
add_tfunc(ne_float_fast, 2, 2, cmp_tfunc, 1)
add_tfunc(lt_float_fast, 2, 2, cmp_tfunc, 1)
add_tfunc(le_float_fast, 2, 2, cmp_tfunc, 1)

    ## checked arithmetic ##
chk_tfunc(@nospecialize(x), @nospecialize(y)) = Tuple{widenconst(x), Bool}
add_tfunc(checked_sadd_int, 2, 2, chk_tfunc, 10)
add_tfunc(checked_uadd_int, 2, 2, chk_tfunc, 10)
add_tfunc(checked_ssub_int, 2, 2, chk_tfunc, 10)
add_tfunc(checked_usub_int, 2, 2, chk_tfunc, 10)
add_tfunc(checked_smul_int, 2, 2, chk_tfunc, 10)
add_tfunc(checked_umul_int, 2, 2, chk_tfunc, 10)
    ## other, misc intrinsics ##
add_tfunc(Core.Intrinsics.llvmcall, 3, INT_INF,
          (@nospecialize(fptr), @nospecialize(rt), @nospecialize(at), a...) -> instanceof_tfunc(rt)[1], 10)
cglobal_tfunc(@nospecialize(fptr)) = Ptr{Cvoid}
cglobal_tfunc(@nospecialize(fptr), @nospecialize(t)) = (isType(t) ? Ptr{t.parameters[1]} : Ptr)
cglobal_tfunc(@nospecialize(fptr), t::Const) = (isa(t.val, Type) ? Ptr{t.val} : Ptr)
add_tfunc(Core.Intrinsics.cglobal, 1, 2, cglobal_tfunc, 5)
add_tfunc(Core.Intrinsics.select_value, 3, 3,
    function (@nospecialize(cnd), @nospecialize(x), @nospecialize(y))
        if isa(cnd, Const)
            if cnd.val === true
                return x
            elseif cnd.val === false
                return y
            else
                return Bottom
            end
        end
        (Bool ⊑ cnd) || return Bottom
        return tmerge(x, y)
    end, 1)
add_tfunc(===, 2, 2,
    function (@nospecialize(x), @nospecialize(y))
        if isa(x, Const) && isa(y, Const)
            return Const(x.val === y.val)
        elseif typeintersect(widenconst(x), widenconst(y)) === Bottom
            return Const(false)
        elseif (isa(x, Const) && y === typeof(x.val) && isdefined(y, :instance)) ||
               (isa(y, Const) && x === typeof(y.val) && isdefined(x, :instance))
            return Const(true)
        elseif isa(x, Conditional) && isa(y, Const)
            y.val === false && return Conditional(x.var, x.elsetype, x.vtype)
            y.val === true && return x
            return x
        elseif isa(y, Conditional) && isa(x, Const)
            x.val === false && return Conditional(y.var, y.elsetype, y.vtype)
            x.val === true && return y
        end
        return Bool
    end, 1)
function isdefined_tfunc(args...)
    arg1 = args[1]
    if isa(arg1, Const)
        a1 = typeof(arg1.val)
    else
        a1 = widenconst(arg1)
    end
    if isType(a1)
        return Bool
    end
    a1 = unwrap_unionall(a1)
    if isa(a1, DataType) && !a1.abstract
        if a1 <: Array # TODO update when deprecation is removed
        elseif a1 === Module
            length(args) == 2 || return Bottom
            sym = args[2]
            Symbol <: widenconst(sym) || return Bottom
            if isa(sym, Const) && isa(sym.val, Symbol) && isa(arg1, Const) && isdefined(arg1.val, sym.val)
                return Const(true)
            end
        elseif length(args) == 2 && isa(args[2], Const)
            val = args[2].val
            idx::Int = 0
            if isa(val, Symbol)
                idx = fieldindex(a1, val, false)
            elseif isa(val, Int)
                idx = val
            else
                return Bottom
            end
            if 1 <= idx <= a1.ninitialized
                return Const(true)
            elseif a1.name === _NAMEDTUPLE_NAME
                if isleaftype(a1)
                    return Const(false)
                end
            elseif idx <= 0 || (!isvatuple(a1) && idx > fieldcount(a1))
                return Const(false)
            elseif !isvatuple(a1) && isbits(fieldtype(a1, idx))
                return Const(true)
            elseif isa(arg1, Const) && isimmutable((arg1::Const).val)
                return Const(isdefined((arg1::Const).val, idx))
            end
        end
    end
    Bool
end
# TODO change INT_INF to 2 when deprecation is removed
add_tfunc(isdefined, 1, INT_INF, isdefined_tfunc, 1)
_const_sizeof(@nospecialize(x)) = try
    # Constant Vector does not have constant size
    isa(x, Vector) && return Int
    return Const(Core.sizeof(x))
catch
    return Int
end
add_tfunc(Core.sizeof, 1, 1,
          function (@nospecialize(x),)
              isa(x, Const) && return _const_sizeof(x.val)
              isa(x, Conditional) && return _const_sizeof(Bool)
              isconstType(x) && return _const_sizeof(x.parameters[1])
              x !== DataType && isleaftype(x) && return _const_sizeof(x)
              return Int
          end, 0)
old_nfields(@nospecialize x) = length((isa(x,DataType) ? x : typeof(x)).types)
add_tfunc(nfields, 1, 1,
    function (@nospecialize(x),)
        isa(x,Const) && return Const(old_nfields(x.val))
        isa(x,Conditional) && return Const(old_nfields(Bool))
        if isType(x)
            # TODO: remove with deprecation in builtins.c for nfields(::Type)
            isleaftype(x.parameters[1]) && return Const(old_nfields(x.parameters[1]))
        elseif isa(x,DataType) && !x.abstract && !(x.name === Tuple.name && isvatuple(x)) && x !== DataType
            if !(x.name === _NAMEDTUPLE_NAME && !isleaftype(x))
                return Const(length(x.types))
            end
        end
        return Int
    end, 0)
add_tfunc(Core._expr, 1, INT_INF, (args...)->Expr, 100)
add_tfunc(applicable, 1, INT_INF, (@nospecialize(f), args...)->Bool, 100)
add_tfunc(Core.Intrinsics.arraylen, 1, 1, x->Int, 4)
add_tfunc(arraysize, 2, 2, (@nospecialize(a), @nospecialize(d))->Int, 4)
add_tfunc(pointerref, 3, 3,
          function (@nospecialize(a), @nospecialize(i), @nospecialize(align))
              a = widenconst(a)
              if a <: Ptr
                  if isa(a,DataType) && isa(a.parameters[1],Type)
                      return a.parameters[1]
                  elseif isa(a,UnionAll) && !has_free_typevars(a)
                      unw = unwrap_unionall(a)
                      if isa(unw,DataType)
                          return rewrap_unionall(unw.parameters[1], a)
                      end
                  end
              end
              return Any
          end, 4)
add_tfunc(pointerset, 4, 4, (@nospecialize(a), @nospecialize(v), @nospecialize(i), @nospecialize(align)) -> a, 5)

function typeof_tfunc(@nospecialize(t))
    if isa(t, Const)
        return Const(typeof(t.val))
    elseif isa(t, Conditional)
        return Const(Bool)
    elseif isType(t)
        tp = t.parameters[1]
        if !isleaftype(tp)
            return DataType # typeof(Kind::Type)::DataType
        else
            return Const(typeof(tp)) # XXX: this is not necessarily true
        end
    elseif isa(t, DataType)
        if isleaftype(t) || isvarargtype(t)
            return Const(t)
        elseif t === Any
            return DataType
        else
            return Type{<:t}
        end
    elseif isa(t, Union)
        a = widenconst(typeof_tfunc(t.a))
        b = widenconst(typeof_tfunc(t.b))
        return Union{a, b}
    elseif isa(t, TypeVar) && !(Any <: t.ub)
        return typeof_tfunc(t.ub)
    elseif isa(t, UnionAll)
        return rewrap_unionall(widenconst(typeof_tfunc(unwrap_unionall(t))), t)
    else
        return DataType # typeof(anything)::DataType
    end
end
add_tfunc(typeof, 1, 1, typeof_tfunc, 0)
add_tfunc(typeassert, 2, 2,
          function (@nospecialize(v), @nospecialize(t))
              t, isexact = instanceof_tfunc(t)
              t === Any && return v
              if isa(v, Const)
                  if !has_free_typevars(t) && !isa(v.val, t)
                      return Bottom
                  end
                  return v
              elseif isa(v, Conditional)
                  if !(Bool <: t)
                      return Bottom
                  end
                  return v
              end
              return typeintersect(v, t)
          end, 4)
add_tfunc(isa, 2, 2,
          function (@nospecialize(v), @nospecialize(t))
              t, isexact = instanceof_tfunc(t)
              if !has_free_typevars(t)
                  if t === Bottom
                      return Const(false)
                  elseif v ⊑ t
                      if isexact
                          return Const(true)
                      end
                  elseif isa(v, Const) || isa(v, Conditional) || (isleaftype(v) && !iskindtype(v))
                      return Const(false)
                  elseif isexact && typeintersect(v, t) === Bottom
                      if !iskindtype(v) #= subtyping currently intentionally answers this query incorrectly for kinds =#
                          return Const(false)
                      end
                  end
              end
              # TODO: handle non-leaftype(t) by testing against lower and upper bounds
              return Bool
          end, 0)
add_tfunc(<:, 2, 2,
          function (@nospecialize(a), @nospecialize(b))
              a, isexact_a = instanceof_tfunc(a)
              b, isexact_b = instanceof_tfunc(b)
              if !has_free_typevars(a) && !has_free_typevars(b)
                  if a <: b
                      if isexact_b || a === Bottom
                          return Const(true)
                      end
                  else
                      if isexact_a || (b !== Bottom && typeintersect(a, b) === Union{})
                          return Const(false)
                      end
                  end
              end
              return Bool
          end, 0)

function const_datatype_getfield_tfunc(sv, fld)
    if (fld == DATATYPE_NAME_FIELDINDEX ||
            fld == DATATYPE_PARAMETERS_FIELDINDEX ||
            fld == DATATYPE_TYPES_FIELDINDEX ||
            fld == DATATYPE_SUPER_FIELDINDEX ||
            fld == DATATYPE_MUTABLE_FIELDINDEX)
        return AbstractEvalConstant(getfield(sv, fld))
    end
    return nothing
end

getfield_tfunc(@nospecialize(s00), @nospecialize(name), @nospecialize(inbounds)) =
    getfield_tfunc(s00, name)
function getfield_tfunc(@nospecialize(s00), @nospecialize(name))
    if isa(s00, TypeVar)
        s00 = s00.ub
    end
    s = unwrap_unionall(s00)
    if isa(s, Union)
        return tmerge(rewrap(getfield_tfunc(s.a, name),s00),
                      rewrap(getfield_tfunc(s.b, name),s00))
    elseif isa(s, Conditional)
        return Bottom # Bool has no fields
    elseif isa(s, Const) || isconstType(s)
        if !isa(s, Const)
            sv = s.parameters[1]
        else
            sv = s.val
        end
        if isa(name, Const)
            nv = name.val
            if isa(sv, UnionAll)
                if nv === :var || nv === 1
                    return Const(sv.var)
                elseif nv === :body || nv === 2
                    return Const(sv.body)
                end
            elseif isa(sv, DataType)
                t = const_datatype_getfield_tfunc(sv, isa(nv, Symbol) ?
                      fieldindex(DataType, nv, false) : nv)
                t !== nothing && return t
            elseif isa(sv, TypeName)
                fld = isa(nv, Symbol) ? fieldindex(TypeName, nv, false) : nv
                if (fld == TYPENAME_NAME_FIELDINDEX ||
                    fld == TYPENAME_MODULE_FIELDINDEX ||
                    fld == TYPENAME_WRAPPER_FIELDINDEX)
                    return AbstractEvalConstant(getfield(sv, fld))
                end
            end
            if isa(sv, Module) && isa(nv, Symbol)
                return abstract_eval_global(sv, nv)
            end
            if !(isa(nv,Symbol) || isa(nv,Int))
                return Bottom
            end
            if (isa(sv, SimpleVector) || isimmutable(sv)) && isdefined(sv, nv)
                return AbstractEvalConstant(getfield(sv, nv))
            end
        end
        s = typeof(sv)
    end
    if isType(s) || !isa(s, DataType) || s.abstract
        return Any
    end
    if s <: Tuple && name ⊑ Symbol
        return Bottom
    end
    if s <: Module
        if name ⊑ Int
            return Bottom
        end
        return Any
    end
    if s.name === _NAMEDTUPLE_NAME && !isleaftype(s)
        # TODO: better approximate inference
        return Any
    end
    if isempty(s.types)
        return Bottom
    end
    if isa(name, Conditional)
        return Bottom # can't index fields with Bool
    end
    if !isa(name, Const)
        if !(Int <: name || Symbol <: name)
            return Bottom
        end
        if length(s.types) == 1
            return rewrap_unionall(unwrapva(s.types[1]), s00)
        end
        # union together types of all fields
        R = reduce(tmerge, Bottom, map(t -> rewrap_unionall(unwrapva(t), s00), s.types))
        # do the same limiting as the known-symbol case to preserve type-monotonicity
        if isempty(s.parameters)
            return R
        end
        return limit_type_depth(R, MAX_TYPE_DEPTH)
    end
    fld = name.val
    if isa(fld,Symbol)
        fld = fieldindex(s, fld, false)
    end
    if !isa(fld,Int)
        return Bottom
    end
    nf = length(s.types)
    if s <: Tuple && fld >= nf && isvarargtype(s.types[nf])
        return rewrap_unionall(unwrapva(s.types[nf]), s00)
    end
    if fld < 1 || fld > nf
        return Bottom
    end
    if isType(s00) && isleaftype(s00.parameters[1])
        sp = s00.parameters[1]
    elseif isa(s00, Const) && isa(s00.val, DataType)
        sp = s00.val
    else
        sp = nothing
    end
    if sp !== nothing
        t = const_datatype_getfield_tfunc(sp, fld)
        t !== nothing && return t
    end
    R = s.types[fld]
    if isempty(s.parameters)
        return R
    end
    # TODO jb/subtype is this still necessary?
    # conservatively limit the type depth here,
    # since the UnionAll type bound is otherwise incorrect
    # in the current type system
    return rewrap_unionall(limit_type_depth(R, MAX_TYPE_DEPTH), s00)
end
add_tfunc(getfield, 2, 3, getfield_tfunc, 1)
add_tfunc(setfield!, 3, 3, (@nospecialize(o), @nospecialize(f), @nospecialize(v)) -> v, 3)
fieldtype_tfunc(@nospecialize(s0), @nospecialize(name), @nospecialize(inbounds)) =
    fieldtype_tfunc(s0, name)
function fieldtype_tfunc(@nospecialize(s0), @nospecialize(name))
    if s0 === Any || s0 === Type || DataType ⊑ s0 || UnionAll ⊑ s0
        return Type
    end
    # fieldtype only accepts DataType and UnionAll, errors on `Module`
    if isa(s0,Const) && (!(isa(s0.val,DataType) || isa(s0.val,UnionAll)) || s0.val === Module)
        return Bottom
    end
    if s0 == Type{Module} || s0 == Type{Union{}} || isa(s0, Conditional)
        return Bottom
    end

    s = instanceof_tfunc(s0)[1]
    u = unwrap_unionall(s)

    if isa(u,Union)
        return tmerge(rewrap(fieldtype_tfunc(u.a, name),s),
                      rewrap(fieldtype_tfunc(u.b, name),s))
    end

    if !isa(u,DataType) || u.abstract
        return Type
    end
    if u.name === _NAMEDTUPLE_NAME && !isleaftype(u)
        return Type
    end
    ftypes = u.types
    if isempty(ftypes)
        return Bottom
    end

    if !isa(name, Const)
        if !(Int <: name || Symbol <: name)
            return Bottom
        end
        return reduce(tmerge, Bottom,
                      Any[ fieldtype_tfunc(s0, Const(i)) for i = 1:length(ftypes) ])
    end

    fld = name.val
    if isa(fld,Symbol)
        fld = fieldindex(u, fld, false)
    end
    if !isa(fld, Int)
        return Bottom
    end
    nf = length(ftypes)
    if u.name === Tuple.name && fld >= nf && isvarargtype(ftypes[nf])
        ft = unwrapva(ftypes[nf])
    elseif fld < 1 || fld > nf
        return Bottom
    else
        ft = ftypes[fld]
    end

    exact = (isa(s0, Const) || isType(s0)) && !has_free_typevars(s)
    ft = rewrap_unionall(ft,s)
    if exact
        return Const(ft)
    end
    return Type{<:ft}
end
add_tfunc(fieldtype, 2, 3, fieldtype_tfunc, 0)

# TODO: handle e.g. apply_type(T, R::Union{Type{Int32},Type{Float64}})
function apply_type_tfunc(@nospecialize(headtypetype), @nospecialize args...)
    if isa(headtypetype, Const)
        headtype = headtypetype.val
    elseif isType(headtypetype) && isleaftype(headtypetype.parameters[1])
        headtype = headtypetype.parameters[1]
    else
        return Any
    end
    largs = length(args)
    if headtype === Union
        largs == 0 && return Const(Bottom)
        largs == 1 && return args[1]
        for i = 1:largs
            ai = args[i]
            if !isa(ai, Const) || !isa(ai.val, Type)
                if !isType(ai)
                    return Any
                end
            end
        end
        ty = Union{}
        allconst = true
        for i = 1:largs
            ai = args[i]
            if isType(ai)
                aty = ai.parameters[1]
                isleaftype(aty) || (allconst = false)
            else
                aty = (ai::Const).val
            end
            ty = Union{ty, aty}
        end
        return allconst ? Const(ty) : Type{ty}
    end
    istuple = (headtype == Tuple)
    if !istuple && !isa(headtype, UnionAll)
        # TODO: return `Bottom` for trying to apply a non-UnionAll
        return Any
    end
    uncertain = false
    canconst = true
    tparams = Any[]
    outervars = Any[]
    for i = 1:largs
        ai = args[i]
        if isType(ai)
            aip1 = ai.parameters[1]
            canconst &= !has_free_typevars(aip1)
            push!(tparams, aip1)
        elseif isa(ai, Const) && (isa(ai.val, Type) || isa(ai.val, TypeVar) || valid_tparam(ai.val))
            push!(tparams, ai.val)
        elseif isa(ai, PartialTypeVar)
            canconst = false
            push!(tparams, ai.tv)
        else
            # TODO: return `Bottom` for trying to apply a non-UnionAll
            uncertain = true
            # These blocks improve type info but make compilation a bit slower.
            # XXX
            #unw = unwrap_unionall(ai)
            #isT = isType(unw)
            #if isT && isa(ai,UnionAll) && contains_is(outervars, ai.var)
            #    ai = rename_unionall(ai)
            #    unw = unwrap_unionall(ai)
            #end
            if istuple
                if i == largs
                    push!(tparams, Vararg)
                # XXX
                #elseif isT
                #    push!(tparams, rewrap_unionall(unw.parameters[1], ai))
                else
                    push!(tparams, Any)
                end
            # XXX
            #elseif isT
            #    push!(tparams, unw.parameters[1])
            #    while isa(ai, UnionAll)
            #        push!(outervars, ai.var)
            #        ai = ai.body
            #    end
            else
                v = TypeVar(:_)
                push!(tparams, v)
                push!(outervars, v)
            end
        end
    end
    local appl
    try
        appl = apply_type(headtype, tparams...)
    catch ex
        # type instantiation might fail if one of the type parameters
        # doesn't match, which could happen if a type estimate is too coarse
        return Type{<:headtype}
    end
    !uncertain && canconst && return Const(appl)
    if isvarargtype(headtype)
        return Type
    end
    if uncertain && type_too_complex(appl, MAX_TYPE_DEPTH)
        return Type{<:headtype}
    end
    if istuple
        return Type{<:appl}
    end
    ans = Type{appl}
    for i = length(outervars):-1:1
        ans = UnionAll(outervars[i], ans)
    end
    return ans
end
add_tfunc(apply_type, 1, INT_INF, apply_type_tfunc, 10)

function invoke_tfunc(@nospecialize(f), @nospecialize(types), @nospecialize(argtype), sv::InferenceState)
    if !isleaftype(Type{types})
        return Any
    end
    argtype = typeintersect(types,limit_tuple_type(argtype, sv.params))
    if argtype === Bottom
        return Bottom
    end
    ft = type_typeof(f)
    types = rewrap_unionall(Tuple{ft, unwrap_unionall(types).parameters...}, types)
    argtype = Tuple{ft, argtype.parameters...}
    entry = ccall(:jl_gf_invoke_lookup, Any, (Any, UInt), types, sv.params.world)
    if entry === nothing
        return Any
    end
    meth = entry.func
    (ti, env) = ccall(:jl_type_intersection_with_env, Any, (Any, Any), argtype, meth.sig)::SimpleVector
    rt, edge = typeinf_edge(meth::Method, ti, env, sv)
    edge !== nothing && add_backedge!(edge::MethodInstance, sv)
    return rt
end

function tuple_tfunc(@nospecialize(argtype))
    if isa(argtype, DataType) && argtype.name === Tuple.name
        p = Vector{Any}()
        for x in argtype.parameters
            if isType(x) && !isa(x.parameters[1], TypeVar)
                xparam = x.parameters[1]
                if isleaftype(xparam) || xparam === Bottom
                    push!(p, typeof(xparam))
                else
                    push!(p, Type)
                end
            else
                push!(p, x)
            end
        end
        t = Tuple{p...}
        # replace a singleton type with its equivalent Const object
        isdefined(t, :instance) && return Const(t.instance)
        return t
    end
    return argtype
end

function builtin_tfunction(@nospecialize(f), argtypes::Array{Any,1},
                           sv::Union{InferenceState,Nothing}, params::Params = sv.params)
    isva = !isempty(argtypes) && isvarargtype(argtypes[end])
    if f === tuple
        for a in argtypes
            if !isa(a, Const)
                return tuple_tfunc(limit_tuple_depth(params, argtypes_to_type(argtypes)))
            end
        end
        return Const(tuple(anymap(a->a.val, argtypes)...))
    elseif f === svec
        return SimpleVector
    elseif f === arrayset
        if length(argtypes) < 4
            isva && return Any
            return Bottom
        end
        return argtypes[2]
    elseif f === arrayref
        if length(argtypes) < 3
            isva && return Any
            return Bottom
        end
        a = widenconst(argtypes[2])
        if a <: Array
            if isa(a, DataType) && (isa(a.parameters[1], Type) || isa(a.parameters[1], TypeVar))
                # TODO: the TypeVar case should not be needed here
                a = a.parameters[1]
                return isa(a, TypeVar) ? a.ub : a
            elseif isa(a, UnionAll) && !has_free_typevars(a)
                unw = unwrap_unionall(a)
                if isa(unw, DataType)
                    return rewrap_unionall(unw.parameters[1], a)
                end
            end
        end
        return Any
    elseif f === Expr
        if length(argtypes) < 1 && !isva
            return Bottom
        end
        return Expr
    elseif f === invoke
        if length(argtypes)>1 && isa(argtypes[1], Const)
            af = argtypes[1].val
            sig = argtypes[2]
            if isa(sig, Const)
                sigty = sig.val
            elseif isType(sig)
                sigty = sig.parameters[1]
            else
                sigty = nothing
            end
            if isa(sigty, Type) && sigty <: Tuple && sv !== nothing
                return invoke_tfunc(af, sigty, argtypes_to_type(argtypes[3:end]), sv)
            end
        end
        return Any
    end
    if isva
        return Any
    end
    if isa(f, IntrinsicFunction)
        if is_pure_intrinsic_infer(f) && all(a -> isa(a, Const), argtypes)
            argvals = anymap(a -> a.val, argtypes)
            try
                return Const(f(argvals...))
            end
        end
        iidx = Int(reinterpret(Int32, f::IntrinsicFunction)) + 1
        if iidx < 0 || iidx > length(T_IFUNC)
            # invalid intrinsic
            return Any
        end
        tf = T_IFUNC[iidx]
    else
        fidx = findfirst(x->x===f, T_FFUNC_KEY)
        if fidx == 0
            # unknown/unhandled builtin function
            return Any
        end
        tf = T_FFUNC_VAL[fidx]
    end
    tf = tf::Tuple{Int, Int, Any}
    if !(tf[1] <= length(argtypes) <= tf[2])
        # wrong # of args
        return Bottom
    end
    return tf[3](argtypes...)
end
