--------------------------------------------------------------------------------
-- luann_af.lua (UPDATED)
-- Comprehensive refactored LuaNN neural network library using luajit_af (ArrayFire)
-- with additional functionality ported from the original luann.lua.
-- PLACE THIS UPDATED CODE IN YOUR luann_af.lua FILE.
--------------------------------------------------------------------------------
local ffi = require("ffi")
local af = require("luajit_af")

-- Define our own af.af_gt if not provided.
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

--------------------------------------------------------------------------------
-- Utility Functions
--------------------------------------------------------------------------------
local function constantLike(x, val)
    local d0, d1, d2, d3 = af.af_get_dims(x)
    local dims = ffi.new("int64_t[4]", d0, d1, d2, d3)
    return af.af_constant(nil, val, 4, dims, af.f32)
end

--------------------------------------------------------------------------------
-- Vectorized Activation Functions
--------------------------------------------------------------------------------
local activations = {}

function activations.sigmoid(x)
    local neg_x = af.af_mul_scalar(x, -1)
    local exp_neg = af.af_exp(neg_x)
    local one = constantLike(x, 1)
    local denom = af.af_add(exp_neg, one, false)
    return af.af_rdiv_scalar(1, denom)
end

function activations.sigmoid_derivative(x)
    local s = activations.sigmoid(x)
    local one = constantLike(x, 1)
    local diff = af.af_sub(one, s, false)
    return af.af_mul(s, diff, false)
end

function activations.relu(x)
    local absx = af.af_abs(x)
    return af.af_mul_scalar(af.af_add(x, absx, false), 0.5)
end

function activations.relu_derivative(x)
    local zero = constantLike(x, 0)
    return af.af_gt(x, zero)
end

function activations.tanh(x)
    local two = constantLike(x, 2)
    local two_x = af.af_mul(x, two, false)
    local exp_two_x = af.af_exp(two_x)
    local one = constantLike(x, 1)
    local numerator = af.af_sub(exp_two_x, one, false)
    local denominator = af.af_add(exp_two_x, one, false)
    return af.af_div(numerator, denominator, false)
end

function activations.softmax(x)
    local maxVal, _, _ = af.af_max_all(x)
    local maxArr = constantLike(x, maxVal)
    local shifted = af.af_sub(x, maxArr, false)
    local expShifted = af.af_exp(shifted)
    local sumExp, _, _ = af.af_sum_all(expShifted)
    local sumArr = constantLike(expShifted, sumExp)
    return af.af_div(expShifted, sumArr, false)
end

function activations.leakyRelu(x, alpha)
    alpha = alpha or 0.01
    local zero = constantLike(x, 0)
    local mask = af.af_gt(x, zero)
    local one = constantLike(x, 1)
    local inv_mask = af.af_sub(one, mask, false)
    local pos = af.af_mul(x, mask, false)
    local neg = af.af_mul_scalar(af.af_mul(x, inv_mask, false), alpha, false)
    return af.af_add(pos, neg, false)
end

function activations.leakyRelu_derivative(x, alpha)
    alpha = alpha or 0.01
    local zero = constantLike(x, 0)
    local mask = af.af_gt(x, zero)
    local one = constantLike(x, 1)
    local inv_mask = af.af_sub(one, mask, false)
    local pos = mask
    local neg = af.af_mul_scalar(inv_mask, alpha, false)
    return af.af_add(pos, neg, false)
end

function activations.elu(x, alpha)
    alpha = alpha or 1.0
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

function activations.elu_derivative(x, alpha)
    alpha = alpha or 1.0
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

function activations.selu(x, lambda, alpha)
    lambda = lambda or 1.0507
    alpha = alpha or 1.67326
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

function activations.selu_derivative(x, lambda, alpha)
    lambda = lambda or 1.0507
    alpha = alpha or 1.67326
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

function activations.swish(x, beta)
    beta = beta or 1.0
    local beta_x = af.af_mul_scalar(x, beta, false)
    local sig = activations.sigmoid(beta_x)
    return af.af_mul(x, sig, false)
end

function activations.swish_derivative(x, beta)
    beta = beta or 1.0
    local beta_x = af.af_mul_scalar(x, beta, false)
    local sig = activations.sigmoid(beta_x)
    local one = constantLike(x, 1)
    local inv_sig = af.af_sub(one, sig, false)
    local term = af.af_mul(x, sig, false)
    term = af.af_mul_scalar(term, beta, false)
    term = af.af_mul(term, inv_sig, false)
    return af.af_add(sig, term, false)
end

--------------------------------------------------------------------------------
-- Weight Initialization Strategies
--------------------------------------------------------------------------------
local weightInits = {}

function weightInits.default(inputSize, outputSize, dims)
    local dims_c = ffi.new("int64_t[?]", #dims)
    for i = 0, #dims - 1 do
        dims_c[i] = dims[i+1]
    end
    return af.af_randu(2, dims_c, af.f32)
end

function weightInits.xavier(inputSize, outputSize, dims)
    local dims_c = ffi.new("int64_t[?]", #dims)
    for i = 0, #dims - 1 do
        dims_c[i] = dims[i+1]
    end
    local W = af.af_randn(2, dims_c, af.f32)
    local scale = math.sqrt(2 / (inputSize + outputSize))
    return af.af_mul_scalar(W, scale)
end

--------------------------------------------------------------------------------
-- Attention Module (Scaled Dot-Product & Multi-Head)
--------------------------------------------------------------------------------
local Attention = {}

function Attention.scaledDotProduct(query, key, value, dropoutRate)
    local _, d = af.af_get_dims(key)
    local keyT = af.af_transpose(key, false)
    local scores = af.af_matmul(query, keyT, 0, 0)
    scores = af.af_div_scalar(scores, math.sqrt(d))
    local weights = activations.softmax(scores)
    -- Dropout not implemented in ArrayFire version.
    return af.af_matmul(weights, value, 0, 0)
end

function Attention.splitHead(matrix, headIndex, headSize)
    local m, d, _, _ = af.af_get_dims(matrix)
    local startCol = (headIndex - 1) * headSize
    local endCol = headIndex * headSize - 1
    local rowSeq = { isSeq = true, seq = ffi.new("af_seq", {begin = 0, ["end"] = m - 1, step = 1, is_gfor = 0}) }
    local colSeq = { isSeq = true, seq = ffi.new("af_seq", {begin = startCol, ["end"] = endCol, step = 1, is_gfor = 0}) }
    local indices = { rowSeq, colSeq }
    local out, err = af.af_index_gen(matrix, indices)
    return out
end

function Attention.concatHeads(heads, dModel)
    local result = heads[1]
    for i = 2, #heads do
        result = af.af_join(1, result, heads[i])
    end
    return result
end

function Attention.multiHead(query, key, value, numHeads, dropoutRate)
    local m, dModel, _, _ = af.af_get_dims(query)
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
-- Layer Structure (Fully Connected / Attention)
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
    for i = 0, #biasDims - 1 do
        dims_bias[i] = biasDims[i+1]
    end
    self.b = af.af_constant(nil, 0, 2, dims_bias, af.f32)
    self.dropoutRate = dropoutRate or 0
    self.isAttentionLayer = isAttentionLayer or false
    if self.isAttentionLayer then
        self.attentionHeads = numCells
    end
    self.activation = "sigmoid"
    local dims_mW = ffi.new("int64_t[2]", dims[1], dims[2])
    self.mW = af.af_constant(nil, 0, 2, dims_mW, af.f32)
    self.vW = af.af_constant(nil, 0, 2, dims_mW, af.f32)
    local dims_mb = ffi.new("int64_t[2]", biasDims[1], biasDims[2])
    self.mb = af.af_constant(nil, 0, 2, dims_mb, af.f32)
    self.vb = af.af_constant(nil, 0, 2, dims_mb, af.f32)
    self.timestep = 0
    return self
end

function Layer:applyDropout(a, isTraining)
    if isTraining and self.dropoutRate > 0 then
        return af.af_mul_scalar(a, 1 - self.dropoutRate)
    end
    return a
end

--------------------------------------------------------------------------------
-- Neural Network (LuaNN) Module
--------------------------------------------------------------------------------
local Network = {}
Network.__index = Network

function Network:new(configuration)
    local self = setmetatable({}, Network)
    self.learningRate = configuration.learningRate or 0.01
    self.l1Lambda = configuration.l1Lambda or 0
    self.l2Lambda = configuration.l2Lambda or 0
    self.layers = {}
    local numLayers = #configuration.layers - 1
    for i = 2, #configuration.layers do
        local prevLayer = configuration.layers[i-1]
        local inSize
        if type(prevLayer) == "table" then
            inSize = prevLayer.numCells or prevLayer.heads or 0
        else
            inSize = prevLayer
        end
        local outConfig = configuration.layers[i]
        local outSize, isAttention
        if type(outConfig) == "table" then
            outSize = outConfig.numCells or outConfig.heads or 0
            isAttention = outConfig.isAttention or false
        else
            outSize = outConfig
            isAttention = false
        end
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
    self.input = af.af_create_array(inputs, 1, {#inputs}, af.f32)
end

function Network:updateLearningRate(newLearningRate)
    self.learningRate = newLearningRate
end

function Network:forward()
    local a = self.input
    self.zs = {}
    self.as = {a}
    for i, layer in ipairs(self.layers) do
        local z = af.af_matmul(layer.W, a, 0, 0)
        z = af.af_add(z, layer.b, false)
        self.zs[i] = z
        if layer.isAttentionLayer then
            a = Attention.multiHead(z, z, z, layer.attentionHeads, layer.dropoutRate)
        else
            if layer.activation == "softmax" then
                a = activations.softmax(z)
            elseif layer.activation == "relu" then
                a = activations.relu(z)
            elseif layer.activation == "tanh" then
                a = activations.tanh(z)
            elseif activations[layer.activation] then
                a = activations[layer.activation](z)
            else
                a = activations.sigmoid(z)
            end
        end
        a = layer:applyDropout(a, true)
        self.as[i+1] = a
    end
    return a
end

function Network:calculateGoodness(a)
    local sq = af.af_mul(a, a, false)
    local sum, _, _ = af.af_sum_all(sq)
    return sum
end

function Network:getSignals(layerIndex)
    local a = self.as[layerIndex]
    local data_ptr, numElems = af.af_get_data_ptr(a)
    local result = {}
    for i = 0, numElems - 1 do
        result[i+1] = data_ptr[i]
    end
    return result
end

--------------------------------------------------------------------------------
-- Backpropagation with AdamW and Regularization
--------------------------------------------------------------------------------
function Network:backpropagate(targetOutputs, activationFuncs, adamParams)
    self:forward()
    local L = #self.layers
    local deltas = {}
    local target = af.af_create_array(targetOutputs, 1, {#targetOutputs}, af.f32)
    local lastLayer = self.layers[L]
    if lastLayer.activation == "softmax" then
        deltas[L] = af.af_sub(self.as[L+1], target, false)
    else
        local diff = af.af_sub(self.as[L+1], target, false)
        local sp = activations.sigmoid_derivative(self.zs[L])
        deltas[L] = af.af_mul(diff, sp, false)
    end
    for i = L - 1, 1, -1 do
        local nextLayer = self.layers[i+1]
        local W_next_T = af.af_transpose(nextLayer.W, false)
        local temp = af.af_matmul(W_next_T, deltas[i+1], 0, 0)
        local sp = activations.sigmoid_derivative(self.zs[i])
        deltas[i] = af.af_mul(temp, sp, false)
    end
    for i = 1, L do
        local layer = self.layers[i]
        local a_prev_T = af.af_transpose(self.as[i], false)
        local dW = af.af_matmul(deltas[i], a_prev_T, 0, 0)
        local db = deltas[i]
        if self.l2Lambda ~= 0 then
            local reg = af.af_mul_scalar(layer.W, self.l2Lambda, false)
            dW = af.af_add(dW, reg, false)
        end
        if self.l1Lambda ~= 0 then
            local absW = af.af_abs(layer.W)
            local signW = af.af_div(layer.W, absW, false)
            local reg1 = af.af_mul_scalar(signW, self.l1Lambda, false)
            dW = af.af_add(dW, reg1, false)
        end
        if adamParams then
            layer.timestep = layer.timestep + 1
            local beta1 = adamParams.beta1 or 0.9
            local beta2 = adamParams.beta2 or 0.999
            local epsilon = adamParams.epsilon or 1e-8
            layer.mW = af.af_add(af.af_mul_scalar(layer.mW, beta1), af.af_mul_scalar(dW, (1-beta1), false), false)
            local dW_squared = af.af_mul(dW, dW, false)
            layer.vW = af.af_add(af.af_mul_scalar(layer.vW, beta2), af.af_mul_scalar(dW_squared, (1-beta2), false), false)
            local mW_hat = af.af_div_scalar(layer.mW, (1 - math.pow(beta1, layer.timestep)))
            local vW_hat = af.af_div_scalar(layer.vW, (1 - math.pow(beta2, layer.timestep)))
            local updateW = af.af_div(af.af_mul_scalar(mW_hat, self.learningRate), af.af_add(af.af_sqrt(vW_hat), constantLike(vW_hat, epsilon), false), false)
            layer.W = af.af_sub(layer.W, updateW, false)
            
            layer.mb = af.af_add(af.af_mul_scalar(layer.mb, beta1), af.af_mul_scalar(db, (1-beta1), false), false)
            local db_squared = af.af_mul(db, db, false)
            layer.vb = af.af_add(af.af_mul_scalar(layer.vb, beta2), af.af_mul_scalar(db_squared, (1-beta2), false), false)
            local mb_hat = af.af_div_scalar(layer.mb, (1 - math.pow(beta1, layer.timestep)))
            local vb_hat = af.af_div_scalar(layer.vb, (1 - math.pow(beta2, layer.timestep)))
            local update_b = af.af_div(af.af_mul_scalar(mb_hat, self.learningRate), af.af_add(af.af_sqrt(vb_hat), constantLike(vb_hat, epsilon), false), false)
            layer.b = af.af_sub(layer.b, update_b, false)
        else
            layer.W = af.af_sub(layer.W, af.af_mul_scalar(dW, self.learningRate, false), false)
            layer.b = af.af_sub(layer.b, af.af_mul_scalar(db, self.learningRate, false), false)
        end
    end
end

--------------------------------------------------------------------------------
-- Forward-Forward Learning Integration
--------------------------------------------------------------------------------
function Network:forwardForward(activationFuncs, data, isPositive)
    self.as = {self.input}
    self.zs = {}
    local goodness = {}
    local a = self.input
    for i, layer in ipairs(self.layers) do
        local z = af.af_matmul(layer.W, a, 0, 0)
        z = af.af_add(z, layer.b, false)
        self.zs[i] = z
        if layer.isAttentionLayer then
            a = Attention.multiHead(z, z, z, layer.attentionHeads, layer.dropoutRate)
        else
            if layer.activation == "softmax" then
                a = activations.softmax(z)
            elseif layer.activation == "relu" then
                a = activations.relu(z)
            elseif layer.activation == "tanh" then
                a = activations.tanh(z)
            elseif activations[layer.activation] then
                a = activations[layer.activation](z)
            else
                a = activations.sigmoid(z)
            end
        end
        a = layer:applyDropout(a, true)
        self.as[i+1] = a
        goodness[i] = self:calculateGoodness(a)
        local a_prev_T = af.af_transpose(self.as[i], false)
        local sign = isPositive and -1 or 1
        local deltaW = af.af_mul_scalar(af.af_matmul(a, a_prev_T, 0, 0), self.learningRate * sign, false)
        layer.W = af.af_sub(layer.W, deltaW, false)
    end
    return goodness
end

--------------------------------------------------------------------------------
-- Backpropagation with Attention Handling
--------------------------------------------------------------------------------
function Network:backpropagateWithAttention(targetOutputs, activationFuncs, adamParams)
    self:forward()
    local L = #self.layers
    local deltas = {}
    local target = af.af_create_array(targetOutputs, 1, {#targetOutputs}, af.f32)
    local lastLayer = self.layers[L]
    if lastLayer.activation == "softmax" then
        deltas[L] = af.af_sub(self.as[L+1], target, false)
    else
        local diff = af.af_sub(self.as[L+1], target, false)
        local sp = activations.sigmoid_derivative(self.zs[L])
        deltas[L] = af.af_mul(diff, sp, false)
    end
    for i = L - 1, 1, -1 do
        local nextLayer = self.layers[i+1]
        local W_next_T = af.af_transpose(nextLayer.W, false)
        local temp = af.af_matmul(W_next_T, deltas[i+1], 0, 0)
        local sp = activations.sigmoid_derivative(self.zs[i])
        deltas[i] = af.af_mul(temp, sp, false)
    end
    for i = 1, L do
        local layer = self.layers[i]
        local a_prev_T = af.af_transpose(self.as[i], false)
        local dW = af.af_matmul(deltas[i], a_prev_T, 0, 0)
        local db = deltas[i]
        if layer.isAttentionLayer then
            layer.W = af.af_sub(layer.W, af.af_mul_scalar(dW, self.learningRate, false), false)
            layer.b = af.af_sub(layer.b, af.af_mul_scalar(db, self.learningRate, false), false)
        else
            if adamParams then
                layer.timestep = layer.timestep + 1
                local beta1 = adamParams.beta1 or 0.9
                local beta2 = adamParams.beta2 or 0.999
                local epsilon = adamParams.epsilon or 1e-8
                layer.mW = af.af_add(af.af_mul_scalar(layer.mW, beta1), af.af_mul_scalar(dW, (1-beta1), false), false)
                local dW_squared = af.af_mul(dW, dW, false)
                layer.vW = af.af_add(af.af_mul_scalar(layer.vW, beta2), af.af_mul_scalar(dW_squared, (1-beta2), false), false)
                local mW_hat = af.af_div_scalar(layer.mW, (1 - math.pow(beta1, layer.timestep)))
                local vW_hat = af.af_div_scalar(layer.vW, (1 - math.pow(beta2, layer.timestep)))
                local updateW = af.af_div(af.af_mul_scalar(mW_hat, self.learningRate), af.af_add(af.af_sqrt(vW_hat), constantLike(vW_hat, epsilon), false), false)
                layer.W = af.af_sub(layer.W, updateW, false)
                
                layer.mb = af.af_add(af.af_mul_scalar(layer.mb, beta1), af.af_mul_scalar(db, (1-beta1), false), false)
                local db_squared = af.af_mul(db, db, false)
                layer.vb = af.af_add(af.af_mul_scalar(layer.vb, beta2), af.af_mul_scalar(db_squared, (1-beta2), false), false)
                local mb_hat = af.af_div_scalar(layer.mb, (1 - math.pow(beta1, layer.timestep)))
                local vb_hat = af.af_div_scalar(layer.vb, (1 - math.pow(beta2, layer.timestep)))
                local update_b = af.af_div(af.af_mul_scalar(mb_hat, self.learningRate), af.af_add(af.af_sqrt(vb_hat), constantLike(vb_hat, epsilon), false), false)
                layer.b = af.af_sub(layer.b, update_b, false)
            else
                layer.W = af.af_sub(layer.W, af.af_mul_scalar(dW, self.learningRate, false), false)
                layer.b = af.af_sub(layer.b, af.af_mul_scalar(db, self.learningRate, false), false)
            end
        end
    end
end

--------------------------------------------------------------------------------
-- LSTM Cell Implementation using ArrayFire
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
    for i = 0, #biasDims - 1 do
        dims_bias[i] = biasDims[i+1]
    end
    self.bi = af.af_constant(nil, 0, 2, dims_bias, af.f32)
    self.bf = af.af_constant(nil, 0, 2, dims_bias, af.f32)
    self.bc = af.af_constant(nil, 0, 2, dims_bias, af.f32)
    self.bo = af.af_constant(nil, 0, 2, dims_bias, af.f32)
    local dims_state = ffi.new("int64_t[2]", outputSize, 1)
    self.stateC = af.af_constant(nil, 0, 2, dims_state, af.f32)
    self.stateH = af.af_constant(nil, 0, 2, dims_state, af.f32)
    return self
end

function LSTMCell:forward(inputs)
    local hPrev = self.stateH
    local cPrev = self.stateC
    local Wi_x = af.af_matmul(self.Wi, inputs, 0, 0)
    local Ui_h = af.af_matmul(self.Ui, hPrev, 0, 0)
    local i_gate = activations.sigmoid(af.af_add(af.af_add(Wi_x, Ui_h, false), self.bi, false))
    local Wf_x = af.af_matmul(self.Wf, inputs, 0, 0)
    local Uf_h = af.af_matmul(self.Uf, hPrev, 0, 0)
    local f_gate = activations.sigmoid(af.af_add(af.af_add(Wf_x, Uf_h, false), self.bf, false))
    local Wc_x = af.af_matmul(self.Wc, inputs, 0, 0)
    local Uc_h = af.af_matmul(self.Uc, hPrev, 0, 0)
    local g_gate = activations.tanh(af.af_add(af.af_add(Wc_x, Uc_h, false), self.bc, false))
    local Wo_x = af.af_matmul(self.Wo, inputs, 0, 0)
    local Uo_h = af.af_matmul(self.Uo, hPrev, 0, 0)
    local o_gate = activations.sigmoid(af.af_add(af.af_add(Wo_x, Uo_h, false), self.bo, false))
    local cNew = af.af_add(af.af_mul(f_gate, cPrev, false), af.af_mul(i_gate, g_gate, false), false)
    self.stateC = cNew
    local hNew = af.af_mul(o_gate, activations.tanh(cNew), false)
    self.stateH = hNew
    return hNew
end

--------------------------------------------------------------------------------
-- Module Exports
--------------------------------------------------------------------------------
local luann_af = {}
luann_af.Network = Network
luann_af.Attention = Attention
luann_af.LSTMCell = LSTMCell
luann_af.activations = activations
luann_af.weightInits = weightInits
return luann_af
