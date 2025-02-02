local ffi = require("ffi")
local af = require("luajit_af")


if not af.af_gt then
    function af.af_gt(x, y)
        local data_x, n = af.af_get_data_ptr(x)
        local data_y, _ = af.af_get_data_ptr(y)
        local result = {}
        for i = 0, n - 1 do
            result[i+1] = (data_x[i] > data_y[i]) and 1 or 0
        end
        return af.af_create_array(result, 1, {n}, af.f32)
    end
end

-- Helper: create a real constant array like x.
local function constantLike(x, val)
    local d0, d1, d2, d3 = af.af_get_dims(x)
    local dims = ffi.new("int64_t[4]", d0, d1, d2, d3)
    return af.af_constant(nil, val, 4, dims, af.f32)
end

--------------------------------------------------------------------------------
-- Quaternion math operations (split representation)
--------------------------------------------------------------------------------
local quat = {}

function quat.new(r, i, j, k)
    return {r = r, i = i, j = j, k = k}
end

function quat.constantLike(q, val)
    local r = constantLike(q.r, val)
    local i = constantLike(q.i, 0)
    local j = constantLike(q.j, 0)
    local k = constantLike(q.k, 0)
    return quat.new(r, i, j, k)
end

function quat.add(q1, q2)
    return quat.new(af.af_add(q1.r, q2.r, false),
                    af.af_add(q1.i, q2.i, false),
                    af.af_add(q1.j, q2.j, false),
                    af.af_add(q1.k, q2.k, false))
end

function quat.sub(q1, q2)
    return quat.new(af.af_sub(q1.r, q2.r, false),
                    af.af_sub(q1.i, q2.i, false),
                    af.af_sub(q1.j, q2.j, false),
                    af.af_sub(q1.k, q2.k, false))
end

function quat.mul(q1, q2)
    local r = af.af_sub(af.af_sub(af.af_sub(af.af_mul(q1.r, q2.r, false),
                              af.af_mul(q1.i, q2.i, false), false),
                              af.af_mul(q1.j, q2.j, false), false),
                              af.af_mul(q1.k, q2.k, false), false)
    local i = af.af_add(af.af_add(af.af_mul(q1.r, q2.i, false),
                              af.af_mul(q1.i, q2.r, false), false),
                              af.af_sub(af.af_mul(q1.j, q2.k, false),
                              af.af_mul(q1.k, q2.j, false), false), false)
    local j = af.af_add(af.af_sub(af.af_mul(q1.r, q2.j, false),
                              af.af_mul(q1.i, q2.k, false), false),
                              af.af_add(af.af_mul(q1.j, q2.r, false),
                              af.af_mul(q1.k, q2.i, false), false), false)
    local k = af.af_add(af.af_add(af.af_mul(q1.r, q2.k, false),
                              af.af_mul(q1.i, q2.j, false), false),
                              af.af_sub(af.af_mul(q1.k, q2.r, false),
                              af.af_mul(q1.j, q2.i, false), false), false)
    return quat.new(r, i, j, k)
end

function quat.mul_scalar(q, s)
    return quat.new(af.af_mul_scalar(q.r, s, false),
                    af.af_mul_scalar(q.i, s, false),
                    af.af_mul_scalar(q.j, s, false),
                    af.af_mul_scalar(q.k, s, false))
end

function quat.div_scalar(q, s)
    return quat.new(af.af_div_scalar(q.r, s),
                    af.af_div_scalar(q.i, s),
                    af.af_div_scalar(q.j, s),
                    af.af_div_scalar(q.k, s))
end

function quat.exp(q)
    local vnorm = af.af_sqrt(af.af_add(af.af_add(af.af_mul(q.i, q.i, false),
                     af.af_mul(q.j, q.j, false), false),
                     af.af_mul(q.k, q.k, false), false))
    local exp_a = af.af_exp(q.r)
    local cos_v = af.af_cos(vnorm)
    local sin_v = af.af_sin(vnorm)
    local epsilon = constantLike(q.r, 1e-8)
    local vnorm_safe = af.af_add(vnorm, epsilon, false)
    local factor = af.af_div(sin_v, vnorm_safe, false)
    local r_res = af.af_mul(exp_a, cos_v, false)
    local i_res = af.af_mul(af.af_mul(exp_a, factor, false), q.i, false)
    local j_res = af.af_mul(af.af_mul(exp_a, factor, false), q.j, false)
    local k_res = af.af_mul(af.af_mul(exp_a, factor, false), q.k, false)
    return quat.new(r_res, i_res, j_res, k_res)
end

function quat.matmul(q, a)
    local Qr, Qi, Qj, Qk = q.r, q.i, q.j, q.k
    local Ar, Ai, Aj, Ak = a.r, a.i, a.j, a.k
    local Rr = af.af_sub(af.af_sub(af.af_sub(af.af_matmul(Qr, Ar, 0, 0),
                              af.af_matmul(Qi, Ai, 0, 0), false),
                              af.af_matmul(Qj, Aj, 0, 0), false),
                              af.af_matmul(Qk, Ak, 0, 0), false)
    local Ri = af.af_add(af.af_add(af.af_matmul(Qr, Ai, 0, 0),
                              af.af_matmul(Qi, Ar, 0, 0), false),
                              af.af_sub(af.af_matmul(Qj, Ak, 0, 0),
                              af.af_matmul(Qk, Aj, 0, 0), false), false)
    local Rj = af.af_add(af.af_sub(af.af_matmul(Qr, Aj, 0, 0),
                              af.af_matmul(Qi, Ak, 0, 0), false),
                              af.af_add(af.af_matmul(Qj, Ar, 0, 0),
                              af.af_matmul(Qk, Ai, 0, 0), false), false)
    local Rk = af.af_add(af.af_add(af.af_matmul(Qr, Ak, 0, 0),
                              af.af_mul_scalar(af.af_matmul(Qi, Aj, 0, 0), 1, false), false),
                              af.af_sub(af.af_matmul(Qk, Ar, 0, 0),
                              af.af_matmul(Qj, Ai, 0, 0), false), false)
    return quat.new(Rr, Ri, Rj, Rk)
end

function quat.transpose(q, conjugate)
    if conjugate then
        return quat.new(af.af_transpose(q.r, false),
                        af.af_mul_scalar(af.af_transpose(q.i, false), -1, false),
                        af.af_mul_scalar(af.af_transpose(q.j, false), -1, false),
                        af.af_mul_scalar(af.af_transpose(q.k, false), -1, false))
    else
        return quat.new(af.af_transpose(q.r, false),
                        af.af_transpose(q.i, false),
                        af.af_transpose(q.j, false),
                        af.af_transpose(q.k, false))
    end
end

function quat.hadamard(q1, q2)
    return quat.new(af.af_mul(q1.r, q2.r, false),
                    af.af_mul(q1.i, q2.i, false),
                    af.af_mul(q1.j, q2.j, false),
                    af.af_mul(q1.k, q2.k, false))
end

function quat.square(q)
    return quat.new(af.af_mul(q.r, q.r, false),
                    af.af_mul(q.i, q.i, false),
                    af.af_mul(q.j, q.j, false),
                    af.af_mul(q.k, q.k, false))
end

function quat.sqrt(q)
    return quat.new(af.af_sqrt(q.r),
                    af.af_sqrt(q.i),
                    af.af_sqrt(q.j),
                    af.af_sqrt(q.k))
end

function quat.apply_elementwise(q, func)
    return quat.new(func(q.r), func(q.i), func(q.j), func(q.k))
end

--------------------------------------------------------------------------------
-- Quaternion Activation Functions (split activations)
--------------------------------------------------------------------------------
local qactivations = {}

function qactivations.sigmoid(q)
    local function apply_sigmoid(x)
        local neg_x = af.af_mul_scalar(x, -1, false)
        local exp_neg = af.af_exp(neg_x)
        local one = constantLike(x, 1)
        local denom = af.af_add(exp_neg, one, false)
        return af.af_rdiv_scalar(1, denom)
    end
    return quat.apply_elementwise(q, apply_sigmoid)
end

function qactivations.sigmoid_derivative(q)
    local s = qactivations.sigmoid(q)
    local function deriv(x, sx)
        local one = constantLike(x, 1)
        local diff = af.af_sub(one, sx, false)
        return af.af_mul(sx, diff, false)
    end
    return quat.new(deriv(q.r, s.r), deriv(q.i, s.i), deriv(q.j, s.j), deriv(q.k, s.k))
end

function qactivations.relu(q)
    local function apply_relu(x)
        local absx = af.af_abs(x)
        return af.af_mul_scalar(af.af_add(x, absx, false), 0.5)
    end
    return quat.apply_elementwise(q, apply_relu)
end

function qactivations.relu_derivative(q)
    local function apply_relu_deriv(x)
        local zero = constantLike(x, 0)
        return af.af_gt(x, zero)
    end
    return quat.apply_elementwise(q, apply_relu_deriv)
end

function qactivations.tanh(q)
    local function apply_tanh(x)
        local two = constantLike(x, 2)
        local two_x = af.af_mul(x, two, false)
        local exp_two_x = af.af_exp(two_x)
        local one = constantLike(x, 1)
        local numerator = af.af_sub(exp_two_x, one, false)
        local denominator = af.af_add(exp_two_x, one, false)
        return af.af_div(numerator, denominator, false)
    end
    return quat.apply_elementwise(q, apply_tanh)
end

function qactivations.softmax(q)
    local function apply_softmax(x)
        local maxVal, _, _ = af.af_max_all(x)
        local maxArr = constantLike(x, maxVal)
        local shifted = af.af_sub(x, maxArr, false)
        local expShifted = af.af_exp(shifted)
        local sumExp, _, _ = af.af_sum_all(expShifted)
        local sumArr = constantLike(expShifted, sumExp)
        return af.af_div(expShifted, sumArr, false)
    end
    return quat.apply_elementwise(q, apply_softmax)
end

function qactivations.leakyRelu(q, alpha)
    alpha = alpha or 0.01
    local function apply_leaky(x)
        local zero = constantLike(x, 0)
        local mask = af.af_gt(x, zero)
        local one = constantLike(x, 1)
        local inv_mask = af.af_sub(one, mask, false)
        local pos = af.af_mul(x, mask, false)
        local neg = af.af_mul_scalar(af.af_mul(x, inv_mask, false), alpha, false)
        return af.af_add(pos, neg, false)
    end
    return quat.apply_elementwise(q, apply_leaky)
end

function qactivations.leakyRelu_derivative(q, alpha)
    alpha = alpha or 0.01
    local function apply_leaky_deriv(x)
        local zero = constantLike(x, 0)
        local mask = af.af_gt(x, zero)
        local one = constantLike(x, 1)
        local inv_mask = af.af_sub(one, mask, false)
        local pos = mask
        local neg = af.af_mul_scalar(inv_mask, alpha, false)
        return af.af_add(pos, neg, false)
    end
    return quat.apply_elementwise(q, apply_leaky_deriv)
end

function qactivations.elu(q, alpha)
    alpha = alpha or 1.0
    local function apply_elu(x)
        local zero = constantLike(x, 0)
        local mask = af.af_gt(x, zero)
        local one = constantLike(x, 1)
        local inv_mask = af.af_sub(one, mask, false)
        local pos = af.af_mul(x, mask, false)
        local exp_x = af.af_exp(x)
        local neg = af.af_mul_scalar(af.af_sub(exp_x, one, false), alpha, false)
        neg = af.af_mul(neg, inv_mask, false)
        return af.af_add(pos, neg, false)
    end
    return quat.apply_elementwise(q, apply_elu)
end

function qactivations.elu_derivative(q, alpha)
    alpha = alpha or 1.0
    local function apply_elu_deriv(x)
        local zero = constantLike(x, 0)
        local mask = af.af_gt(x, zero)
        local one = constantLike(x, 1)
        local inv_mask = af.af_sub(one, mask, false)
        local pos = mask
        local exp_x = af.af_exp(x)
        local neg = af.af_mul_scalar(exp_x, alpha, false)
        neg = af.af_mul(neg, inv_mask, false)
        return af.af_add(pos, neg, false)
    end
    return quat.apply_elementwise(q, apply_elu_deriv)
end

function qactivations.selu(q, lambda, alpha)
    lambda = lambda or 1.0507
    alpha = alpha or 1.67326
    local function apply_selu(x)
        local zero = constantLike(x, 0)
        local mask = af.af_gt(x, zero)
        local one = constantLike(x, 1)
        local inv_mask = af.af_sub(one, mask, false)
        local pos = af.af_mul_scalar(x, lambda, false)
        pos = af.af_mul(pos, mask, false)
        local exp_x = af.af_exp(x)
        local neg = af.af_mul_scalar(af.af_sub(exp_x, one, false), lambda * alpha, false)
        neg = af.af_mul(neg, inv_mask, false)
        return af.af_add(pos, neg, false)
    end
    return quat.apply_elementwise(q, apply_selu)
end

function qactivations.selu_derivative(q, lambda, alpha)
    lambda = lambda or 1.0507
    alpha = alpha or 1.67326
    local function apply_selu_deriv(x)
        local zero = constantLike(x, 0)
        local mask = af.af_gt(x, zero)
        local one = constantLike(x, 1)
        local inv_mask = af.af_sub(one, mask, false)
        local pos = af.af_mul_scalar(mask, lambda, false)
        local exp_x = af.af_exp(x)
        local neg = af.af_mul_scalar(exp_x, lambda * alpha, false)
        neg = af.af_mul(neg, inv_mask, false)
        return af.af_add(pos, neg, false)
    end
    return quat.apply_elementwise(q, apply_selu_deriv)
end

function qactivations.swish(q, beta)
    beta = beta or 1.0
    local function apply_swish(x)
        local beta_x = af.af_mul_scalar(x, beta, false)
        local sig = qactivations.sigmoid({r = beta_x, i = beta_x, j = beta_x, k = beta_x}).r
        return af.af_mul(x, sig, false)
    end
    return quat.apply_elementwise(q, apply_swish)
end

function qactivations.swish_derivative(q, beta)
    beta = beta or 1.0
    local function apply_swish_deriv(x)
        local beta_x = af.af_mul_scalar(x, beta, false)
        local sig = qactivations.sigmoid({r = beta_x, i = beta_x, j = beta_x, k = beta_x}).r
        local one = constantLike(x, 1)
        local inv_sig = af.af_sub(one, sig, false)
        local term = af.af_mul(x, sig, false)
        term = af.af_mul_scalar(term, beta, false)
        term = af.af_mul(term, inv_sig, false)
        return af.af_add(sig, term, false)
    end
    return quat.apply_elementwise(q, apply_swish_deriv)
end

--------------------------------------------------------------------------------
-- Weight Initialization Strategies for Quaternions
--------------------------------------------------------------------------------
local weightInits = {}

function weightInits.default(inputSize, outputSize, dims)
    local dims_c = ffi.new("int64_t[?]", #dims)
    for i = 0, #dims - 1 do dims_c[i] = dims[i+1] end
    local r = af.af_randu(2, dims_c, af.f32)
    local i_arr = af.af_randu(2, dims_c, af.f32)
    local j_arr = af.af_randu(2, dims_c, af.f32)
    local k_arr = af.af_randu(2, dims_c, af.f32)
    return quat.new(r, i_arr, j_arr, k_arr)
end

function weightInits.xavier(inputSize, outputSize, dims)
    local dims_c = ffi.new("int64_t[?]", #dims)
    for i = 0, #dims - 1 do dims_c[i] = dims[i+1] end
    local r = af.af_randn(2, dims_c, af.f32)
    local i_arr = af.af_randn(2, dims_c, af.f32)
    local j_arr = af.af_randn(2, dims_c, af.f32)
    local k_arr = af.af_randn(2, dims_c, af.f32)
    local W = quat.new(r, i_arr, j_arr, k_arr)
    local scale = math.sqrt(2 / (inputSize + outputSize))
    return quat.mul_scalar(W, scale)
end

--------------------------------------------------------------------------------
-- Attention Module (Scaled Dot-Product & Multi-Head) for Quaternions
--------------------------------------------------------------------------------
local Attention = {}

function Attention.scaledDotProduct(query, key, value, dropoutRate)
    local _, d, _, _ = af.af_get_dims(key.r)
    local keyT = quat.transpose(key, false)
    local scores = quat.matmul(query, keyT)
    scores = quat.div_scalar(scores, math.sqrt(d))
    local weights = qactivations.softmax(scores)
    return quat.matmul(weights, value)
end

function Attention.splitHead(matrix, headIndex, headSize)
    local m, d, _, _ = af.af_get_dims(matrix.r)
    local startCol = (headIndex - 1) * headSize
    local endCol = headIndex * headSize - 1
    local rowSeq = { isSeq = true, seq = ffi.new("af_seq", {begin = 0, ["end"] = m - 1, step = 1, is_gfor = 0}) }
    local colSeq = { isSeq = true, seq = ffi.new("af_seq", {begin = startCol, ["end"] = endCol, step = 1, is_gfor = 0}) }
    local indices = { rowSeq, colSeq }
    local function index_quat(q)
        local out, err = af.af_index_gen(q, indices)
        return out
    end
    return quat.new(index_quat(matrix.r), index_quat(matrix.i), index_quat(matrix.j), index_quat(matrix.k))
end

function Attention.concatHeads(heads, dModel)
    local function join_field(field)
        local result = heads[1][field]
        for i = 2, #heads do
            result = af.af_join(1, result, heads[i][field])
        end
        return result
    end
    return quat.new(join_field("r"), join_field("i"), join_field("j"), join_field("k"))
end

function Attention.multiHead(query, key, value, numHeads, dropoutRate)
    local m, dModel, _, _ = af.af_get_dims(query.r)
    if dModel < numHeads then
        numHeads = 1
    end
    local headSize = math.floor(dModel / numHeads)
    local heads = {}
    for i = 1, numHeads do
        local qHead = Attention.splitHead(query, i, headSize)
        local kHead = Attention.splitHead(key, i, headSize)
        local vHead = Attention.splitHead(value, i, headSize)
        heads[i] = Attention.scaledDotProduct(qHead, kHead, vHead, dropoutRate)
    end
    return Attention.concatHeads(heads, dModel)
end

--------------------------------------------------------------------------------
-- Layer Structure (Fully Connected / Attention) for Quaternions
--------------------------------------------------------------------------------
local Layer = {}
Layer.__index = Layer

function Layer:new(numInputs, numCells, weightInitMethod, dropoutRate, isAttentionLayer)
    local self = setmetatable({}, Layer)
    local dims = {numCells, numInputs}
    local initFunc = weightInits[weightInitMethod] or weightInits.default
    self.W = initFunc(numInputs, numCells, dims)
    self.numCells = numCells
    local biasDims = {numCells, 1}
    local dims_bias = ffi.new("int64_t[?]", #biasDims)
    for i = 0, #biasDims - 1 do dims_bias[i] = biasDims[i+1] end
    local bias_r = af.af_constant(nil, 0, 2, dims_bias, af.f32)
    local bias_zero = af.af_constant(nil, 0, 2, dims_bias, af.f32)
    self.b = quat.new(bias_r, bias_zero, bias_zero, bias_zero)
    self.dropoutRate = dropoutRate or 0
    self.isAttentionLayer = isAttentionLayer or false
    if self.isAttentionLayer then self.attentionHeads = numCells end
    self.activation = "sigmoid"
    local dims_mW = ffi.new("int64_t[2]", dims[1], dims[2])
    local mW_const = af.af_constant(nil, 0, 2, dims_mW, af.f32)
    self.mW = quat.new(mW_const, mW_const, mW_const, mW_const)
    local dims_mb = ffi.new("int64_t[2]", biasDims[1], biasDims[2])
    local mb_const = af.af_constant(nil, 0, 2, dims_mb, af.f32)
    self.mb = quat.new(mb_const, mb_const, mb_const, mb_const)
    self.vW = quat.new(mW_const, mW_const, mW_const, mW_const)
    self.vb = quat.new(mb_const, mb_const, mb_const, mb_const)
    self.timestep = 0
    return self
end

function Layer:applyDropout(a, isTraining)
    if isTraining and self.dropoutRate > 0 then
        return quat.mul_scalar(a, 1 - self.dropoutRate)
    end
    return a
end

--------------------------------------------------------------------------------
-- Neural Network Module for Quaternions
--------------------------------------------------------------------------------
local Network = {}
Network.__index = Network

function Network:new(configuration)
    local self = setmetatable({}, Network)
    self.learningRate = configuration.learningRate or 0.01
    self.l1Lambda = configuration.l1Lambda or 0
    self.l2Lambda = configuration.l2Lambda or 0
    self.layers = {}
    for i = 2, #configuration.layers do
        local prevLayer = configuration.layers[i-1]
        local inSize = (type(prevLayer) == "table") and (prevLayer.numCells or prevLayer.heads or 0) or prevLayer
        local outConfig = configuration.layers[i]
        local outSize, isAttention = (type(outConfig) == "table") and (outConfig.numCells or outConfig.heads or 0) or outConfig, (type(outConfig)=="table") and outConfig.isAttention or false
        local dropout = (configuration.dropoutRates and configuration.dropoutRates[i-1]) or 0
        local weightInitMethod = configuration.weightInitMethod or "default"
        local layer = Layer:new(inSize, outSize, weightInitMethod, dropout, isAttention)
        if configuration.activations and configuration.activations[i-1] then
            layer.activation = configuration.activations[i-1]
        end
        self.layers[i-1] = layer
    end
    return self
end

function Network:setInputSignals(inputs)
    -- Assume inputs is a quaternion table: {r, i, j, k}
    self.input = inputs
end

function Network:updateLearningRate(newLearningRate)
    self.learningRate = newLearningRate
end

function Network:forward()
    local a = self.input
    self.zs = {}
    self.as = {a}
    for i, layer in ipairs(self.layers) do
        local z = quat.matmul(layer.W, a)
        z = quat.add(z, layer.b)
        self.zs[i] = z
        if layer.isAttentionLayer then
            a = Attention.multiHead(z, z, z, layer.attentionHeads, layer.dropoutRate)
        else
            if layer.activation == "softmax" then
                a = qactivations.softmax(z)
            elseif layer.activation == "relu" then
                a = qactivations.relu(z)
            elseif layer.activation == "tanh" then
                a = qactivations.tanh(z)
            elseif qactivations[layer.activation] then
                a = qactivations[layer.activation](z)
            else
                a = qactivations.sigmoid(z)
            end
        end
        a = layer:applyDropout(a, true)
        self.as[i+1] = a
    end
    return a
end

function Network:calculateGoodness(a)
    local sq = quat.hadamard(a, a)
    local sum_r, _, _ = af.af_sum_all(sq.r)
    return sum_r
end

function Network:getSignals(layerIndex)
    local a = self.as[layerIndex]
    local data_ptr, numElems = af.af_get_data_ptr(a.r)
    local result = {}
    for i = 0, numElems - 1 do result[i+1] = data_ptr[i] end
    return result
end

function Network:backpropagate(targetOutputs, activationFuncs, adamParams)
    self:forward()
    local L = #self.layers
    local deltas = {}
    local target = targetOutputs
    local lastLayer = self.layers[L]
    if lastLayer.activation == "softmax" then
        deltas[L] = quat.sub(self.as[L+1], target)
    else
        local diff = quat.sub(self.as[L+1], target)
        local sp = qactivations.sigmoid_derivative(self.zs[L])
        deltas[L] = quat.hadamard(diff, sp)
    end
    for i = L - 1, 1, -1 do
        local nextLayer = self.layers[i+1]
        local W_next_T = quat.transpose(nextLayer.W, false)
        local temp = quat.matmul(W_next_T, deltas[i+1])
        local sp = qactivations.sigmoid_derivative(self.zs[i])
        deltas[i] = quat.hadamard(temp, sp)
    end
    for i = 1, L do
        local layer = self.layers[i]
        local a_prev_T = quat.transpose(self.as[i], false)
        local dW = quat.matmul(deltas[i], a_prev_T)
        local db = deltas[i]
        if self.l2Lambda ~= 0 then
            local reg = quat.mul_scalar(layer.W, self.l2Lambda)
            dW = quat.add(dW, reg)
        end
        if self.l1Lambda ~= 0 then
            local absW = quat.apply_elementwise(layer.W, af.af_abs)
            local signW = quat.new(af.af_div(layer.W.r, absW.r, false),
                                    af.af_div(layer.W.i, absW.i, false),
                                    af.af_div(layer.W.j, absW.j, false),
                                    af.af_div(layer.W.k, absW.k, false))
            local reg1 = quat.mul_scalar(signW, self.l1Lambda)
            dW = quat.add(dW, reg1)
        end
        if adamParams then
            layer.timestep = layer.timestep + 1
            local beta1 = adamParams.beta1 or 0.9
            local beta2 = adamParams.beta2 or 0.999
            local epsilon = adamParams.epsilon or 1e-8
            layer.mW = quat.add(quat.mul_scalar(layer.mW, beta1), quat.mul_scalar(dW, (1-beta1)))
            local dW_squared = quat.square(dW)
            layer.vW = quat.add(quat.mul_scalar(layer.vW, beta2), quat.mul_scalar(dW_squared, (1-beta2)))
            local mW_hat = quat.div_scalar(layer.mW, (1 - math.pow(beta1, layer.timestep)))
            local vW_hat = quat.div_scalar(layer.vW, (1 - math.pow(beta2, layer.timestep)))
            local updateW = quat.div_scalar(quat.mul_scalar(mW_hat, self.learningRate),
                          af.af_add(af.af_sqrt(vW_hat.r), constantLike(vW_hat.r, epsilon), false))
            layer.W = quat.sub(layer.W, updateW)
            layer.mb = quat.add(quat.mul_scalar(layer.mb, beta1), quat.mul_scalar(db, (1-beta1)))
            local db_squared = quat.square(db)
            layer.vb = quat.add(quat.mul_scalar(layer.vb, beta2), quat.mul_scalar(db_squared, (1-beta2)))
            local mb_hat = quat.div_scalar(layer.mb, (1 - math.pow(beta1, layer.timestep)))
            local vb_hat = quat.div_scalar(layer.vb, (1 - math.pow(beta2, layer.timestep)))
            local update_b = quat.div_scalar(quat.mul_scalar(mb_hat, self.learningRate),
                          af.af_add(af.af_sqrt(vb_hat.r), constantLike(vb_hat.r, epsilon), false))
            layer.b = quat.sub(layer.b, update_b)
        else
            layer.W = quat.sub(layer.W, quat.mul_scalar(dW, self.learningRate))
            layer.b = quat.sub(layer.b, quat.mul_scalar(db, self.learningRate))
        end
    end
end

function Network:forwardForward(activationFuncs, data, isPositive)
    self.as = {self.input}
    self.zs = {}
    local goodness = {}
    local a = self.input
    for i, layer in ipairs(self.layers) do
        local z = quat.matmul(layer.W, a)
        z = quat.add(z, layer.b)
        self.zs[i] = z
        if layer.isAttentionLayer then
            a = Attention.multiHead(z, z, z, layer.attentionHeads, layer.dropoutRate)
        else
            if layer.activation == "softmax" then
                a = qactivations.softmax(z)
            elseif layer.activation == "relu" then
                a = qactivations.relu(z)
            elseif layer.activation == "tanh" then
                a = qactivations.tanh(z)
            elseif qactivations[layer.activation] then
                a = qactivations[layer.activation](z)
            else
                a = qactivations.sigmoid(z)
            end
        end
        a = layer:applyDropout(a, true)
        self.as[i+1] = a
        goodness[i] = self:calculateGoodness(a)
        local a_prev_T = quat.transpose(self.as[i], false)
        local sign = isPositive and -1 or 1
        local deltaW = quat.mul_scalar(quat.matmul(a, a_prev_T), self.learningRate * sign)
        layer.W = quat.sub(layer.W, deltaW)
    end
    return goodness
end

function Network:backpropagateWithAttention(targetOutputs, activationFuncs, adamParams)
    self:forward()
    local L = #self.layers
    local deltas = {}
    local target = targetOutputs
    local lastLayer = self.layers[L]
    if lastLayer.activation == "softmax" then
        deltas[L] = quat.sub(self.as[L+1], target)
    else
        local diff = quat.sub(self.as[L+1], target)
        local sp = qactivations.sigmoid_derivative(self.zs[L])
        deltas[L] = quat.hadamard(diff, sp)
    end
    for i = L - 1, 1, -1 do
        local nextLayer = self.layers[i+1]
        local W_next_T = quat.transpose(nextLayer.W, false)
        local temp = quat.matmul(W_next_T, deltas[i+1])
        local sp = qactivations.sigmoid_derivative(self.zs[i])
        deltas[i] = quat.hadamard(temp, sp)
    end
    for i = 1, L do
        local layer = self.layers[i]
        local a_prev_T = quat.transpose(self.as[i], false)
        local dW = quat.matmul(deltas[i], a_prev_T)
        local db = deltas[i]
        if layer.isAttentionLayer then
            layer.W = quat.sub(layer.W, quat.mul_scalar(dW, self.learningRate))
            layer.b = quat.sub(layer.b, quat.mul_scalar(db, self.learningRate))
        else
            if adamParams then
                layer.timestep = layer.timestep + 1
                local beta1 = adamParams.beta1 or 0.9
                local beta2 = adamParams.beta2 or 0.999
                local epsilon = adamParams.epsilon or 1e-8
                layer.mW = quat.add(quat.mul_scalar(layer.mW, beta1), quat.mul_scalar(dW, (1-beta1)))
                local dW_squared = quat.square(dW)
                layer.vW = quat.add(quat.mul_scalar(layer.vW, beta2), quat.mul_scalar(dW_squared, (1-beta2)))
                local mW_hat = quat.div_scalar(layer.mW, (1 - math.pow(beta1, layer.timestep)))
                local vW_hat = quat.div_scalar(layer.vW, (1 - math.pow(beta2, layer.timestep)))
                local updateW = quat.div_scalar(quat.mul_scalar(mW_hat, self.learningRate),
                              af.af_add(af.af_sqrt(vW_hat.r), constantLike(vW_hat.r, epsilon), false))
                layer.W = quat.sub(layer.W, updateW)
                layer.mb = quat.add(quat.mul_scalar(layer.mb, beta1), quat.mul_scalar(db, (1-beta1)))
                local db_squared = quat.square(db)
                layer.vb = quat.add(quat.mul_scalar(layer.vb, beta2), quat.mul_scalar(db_squared, (1-beta2)))
                local mb_hat = quat.div_scalar(layer.mb, (1 - math.pow(beta1, layer.timestep)))
                local vb_hat = quat.div_scalar(layer.vb, (1 - math.pow(beta2, layer.timestep)))
                local update_b = quat.div_scalar(quat.mul_scalar(mb_hat, self.learningRate),
                              af.af_add(af.af_sqrt(vb_hat.r), constantLike(vb_hat.r, epsilon), false))
                layer.b = quat.sub(layer.b, update_b)
            else
                layer.W = quat.sub(layer.W, quat.mul_scalar(dW, self.learningRate))
                layer.b = quat.sub(layer.b, quat.mul_scalar(db, self.learningRate))
            end
        end
    end
end

--------------------------------------------------------------------------------
-- LSTM Cell Implementation using ArrayFire for Quaternions
--------------------------------------------------------------------------------
local LSTMCell = {}
LSTMCell.__index = LSTMCell

function LSTMCell:new(inputSize, outputSize, weightInitMethod)
    local self = setmetatable({}, LSTMCell)
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.weightInitMethod = weightInitMethod or "default"
    local initFunc = weightInits[self.weightInitMethod] or weightInits.default
    local dims_in = {outputSize, inputSize}
    local dims_h = {outputSize, outputSize}
    self.Wi = initFunc(inputSize, outputSize, dims_in)
    self.Wf = initFunc(inputSize, outputSize, dims_in)
    self.Wc = initFunc(inputSize, outputSize, dims_in)
    self.Wo = initFunc(inputSize, outputSize, dims_in)
    self.Ui = initFunc(outputSize, outputSize, dims_h)
    self.Uf = initFunc(outputSize, outputSize, dims_h)
    self.Uc = initFunc(outputSize, outputSize, dims_h)
    self.Uo = initFunc(outputSize, outputSize, dims_h)
    local biasDims = {outputSize, 1}
    local dims_bias = ffi.new("int64_t[?]", #biasDims)
    for i = 0, #biasDims - 1 do dims_bias[i] = biasDims[i+1] end
    local bias_r = af.af_constant(nil, 0, 2, dims_bias, af.f32)
    local bias_zero = af.af_constant(nil, 0, 2, dims_bias, af.f32)
    self.bi = quat.new(bias_r, bias_zero, bias_zero, bias_zero)
    self.bf = quat.new(bias_r, bias_zero, bias_zero, bias_zero)
    self.bc = quat.new(bias_r, bias_zero, bias_zero, bias_zero)
    self.bo = quat.new(bias_r, bias_zero, bias_zero, bias_zero)
    local dims_state = ffi.new("int64_t[2]", outputSize, 1)
    local state_const = af.af_constant(nil, 0, 2, dims_state, af.f32)
    self.stateC = quat.new(state_const, state_const, state_const, state_const)
    self.stateH = quat.new(state_const, state_const, state_const, state_const)
    return self
end

function LSTMCell:forward(inputs)
    local hPrev = self.stateH
    local cPrev = self.stateC
    local Wi_x = quat.matmul(self.Wi, inputs)
    local Ui_h = quat.matmul(self.Ui, hPrev)
    local i_gate = qactivations.sigmoid(quat.add(quat.add(Wi_x, Ui_h), self.bi))
    local Wf_x = quat.matmul(self.Wf, inputs)
    local Uf_h = quat.matmul(self.Uf, hPrev)
    local f_gate = qactivations.sigmoid(quat.add(quat.add(Wf_x, Uf_h), self.bf))
    local Wc_x = quat.matmul(self.Wc, inputs)
    local Uc_h = quat.matmul(self.Uc, hPrev)
    local g_gate = qactivations.tanh(quat.add(quat.add(Wc_x, Uc_h), self.bc))
    local Wo_x = quat.matmul(self.Wo, inputs)
    local Uo_h = quat.matmul(self.Uo, hPrev)
    local o_gate = qactivations.sigmoid(quat.add(quat.add(Wo_x, Uo_h), self.bo))
    local cNew = quat.add(quat.hadamard(f_gate, cPrev), quat.hadamard(i_gate, g_gate))
    self.stateC = cNew
    local hNew = quat.hadamard(o_gate, qactivations.tanh(cNew))
    self.stateH = hNew
    return hNew
end

--------------------------------------------------------------------------------
-- Module Exports
--------------------------------------------------------------------------------
local luann_af_q = {}
luann_af_q.Network = Network
luann_af_q.Attention = Attention
luann_af_q.LSTMCell = LSTMCell
luann_af_q.activations = qactivations
luann_af_q.weightInits = weightInits
return luann_af_q
