--------------------------------------------------------------------------------
-- test_luann_af_q.lua
-- Comprehensive unit tests for the luann_af_q quaternion neural network library using luaunit
--------------------------------------------------------------------------------

local luaunit = require("luaunit")
local luann_af_q = require("luann_af_q")
local af = require("luajit_af")

-- Helper: Convert an ArrayFire array to a Lua table.
local function toTable(afArray)
    local data_ptr, numElems = af.af_get_data_ptr(afArray)
    local t = {}
    for i = 0, numElems - 1 do
        t[i+1] = data_ptr[i]
    end
    return t
end

-- Helper: Convert a quaternion (with fields r, i, j, k) to a Lua table for each component.
local function toTableQuat(q)
    return {
        r = toTable(q.r),
        i = toTable(q.i),
        j = toTable(q.j),
        k = toTable(q.k)
    }
end

-- Helper: Create a quaternion from a Lua table of numbers.
local function createQuaternion(input)
    local n = #input
    local af_r = af.af_create_array(input, 1, {n}, af.f32)
    local af_i = af.af_create_array(input, 1, {n}, af.f32)
    local af_j = af.af_create_array(input, 1, {n}, af.f32)
    local af_k = af.af_create_array(input, 1, {n}, af.f32)
    return { r = af_r, i = af_i, j = af_j, k = af_k }
end

-- Helper: Create a quaternion from given data with specified order and dimensions.
local function createQuaternionFromData(data, order, dims)
    local af_r = af.af_create_array(data, order, dims, af.f32)
    local af_i = af.af_create_array(data, order, dims, af.f32)
    local af_j = af.af_create_array(data, order, dims, af.f32)
    local af_k = af.af_create_array(data, order, dims, af.f32)
    return { r = af_r, i = af_i, j = af_j, k = af_k }
end

--------------------------------------------------------------------------------
-- Tests for Quaternion Activation Functions
--------------------------------------------------------------------------------
TestActivations = {}

function TestActivations:testSigmoid()
    local input = {0, 1, -1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.sigmoid(q_input)
    local output = toTableQuat(q_output)
    local expected = {0.5, 1/(1+math.exp(-1)), 1/(1+math.exp(1))}
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output.r[i], expected[i], 1e-4)
        luaunit.assertAlmostEquals(output.i[i], expected[i], 1e-4)
        luaunit.assertAlmostEquals(output.j[i], expected[i], 1e-4)
        luaunit.assertAlmostEquals(output.k[i], expected[i], 1e-4)
    end
end

function TestActivations:testSigmoidDerivative()
    local input = {0, 1, -1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.sigmoid_derivative(q_input)
    local output = toTableQuat(q_output)
    for i, x in ipairs(input) do
        local s = 1/(1+math.exp(-x))
        local expected = s * (1 - s)
        luaunit.assertAlmostEquals(output.r[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.i[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.j[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.k[i], expected, 1e-4)
    end
end

function TestActivations:testTanh()
    local input = {0, 1, -1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.tanh(q_input)
    local output = toTableQuat(q_output)
    for i, x in ipairs(input) do
        local expected = math.tanh(x)
        luaunit.assertAlmostEquals(output.r[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.i[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.j[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.k[i], expected, 1e-4)
    end
end

function TestActivations:testSoftmax()
    local input = {1, 2, 3}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.softmax(q_input)
    local output = toTableQuat(q_output)
    local sum_r = 0
    for i = 1, #output.r do
        sum_r = sum_r + output.r[i]
    end
    luaunit.assertAlmostEquals(sum_r, 1, 1e-4)
    -- Also verify one other component.
    local sum_i = 0
    for i = 1, #output.i do
        sum_i = sum_i + output.i[i]
    end
    luaunit.assertAlmostEquals(sum_i, 1, 1e-4)
end

function TestActivations:testRelu()
    local input = {-1, 0, 1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.relu(q_input)
    local output = toTableQuat(q_output)
    for i, x in ipairs(input) do
        local expected = math.max(0, x)
        luaunit.assertAlmostEquals(output.r[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.i[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.j[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.k[i], expected, 1e-4)
    end
end

function TestActivations:testReluDerivative()
    local input = {-1, 0, 1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.relu_derivative(q_input)
    local output = toTableQuat(q_output)
    local expected = {0, 0, 1}
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output.r[i], expected[i], 1e-4)
        luaunit.assertAlmostEquals(output.i[i], expected[i], 1e-4)
        luaunit.assertAlmostEquals(output.j[i], expected[i], 1e-4)
        luaunit.assertAlmostEquals(output.k[i], expected[i], 1e-4)
    end
end

function TestActivations:testLeakyRelu()
    local alpha = 0.1
    local input = {-1, 0, 1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.leakyRelu(q_input, alpha)
    local output = toTableQuat(q_output)
    for i, x in ipairs(input) do
        local expected = (x > 0) and x or alpha * x
        luaunit.assertAlmostEquals(output.r[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.i[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.j[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.k[i], expected, 1e-4)
    end
end

function TestActivations:testLeakyReluDerivative()
    local alpha = 0.1
    local input = {-1, 0, 1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.leakyRelu_derivative(q_input, alpha)
    local output = toTableQuat(q_output)
    for i, x in ipairs(input) do
        local expected = (x > 0) and 1 or alpha
        luaunit.assertAlmostEquals(output.r[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.i[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.j[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.k[i], expected, 1e-4)
    end
end

function TestActivations:testElu()
    local alpha = 1.0
    local input = {-1, 0, 1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.elu(q_input, alpha)
    local output = toTableQuat(q_output)
    for i, x in ipairs(input) do
        local expected = (x > 0) and x or alpha * (math.exp(x) - 1)
        luaunit.assertAlmostEquals(output.r[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.i[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.j[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.k[i], expected, 1e-4)
    end
end

function TestActivations:testEluDerivative()
    local alpha = 1.0
    local input = {-1, 0, 1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.elu_derivative(q_input, alpha)
    local output = toTableQuat(q_output)
    for i, x in ipairs(input) do
        local expected = (x > 0) and 1 or alpha * math.exp(x)
        luaunit.assertAlmostEquals(output.r[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.i[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.j[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.k[i], expected, 1e-4)
    end
end

function TestActivations:testSelu()
    local lambda = 1.0507
    local alpha = 1.67326
    local input = {-1, 0, 1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.selu(q_input, lambda, alpha)
    local output = toTableQuat(q_output)
    for i, x in ipairs(input) do
        local expected = (x > 0) and lambda * x or lambda * alpha * (math.exp(x) - 1)
        luaunit.assertAlmostEquals(output.r[i], expected, 1e-3)
        luaunit.assertAlmostEquals(output.i[i], expected, 1e-3)
        luaunit.assertAlmostEquals(output.j[i], expected, 1e-3)
        luaunit.assertAlmostEquals(output.k[i], expected, 1e-3)
    end
end

function TestActivations:testSeluDerivative()
    local lambda = 1.0507
    local alpha = 1.67326
    local input = {-1, 0, 1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.selu_derivative(q_input, lambda, alpha)
    local output = toTableQuat(q_output)
    for i, x in ipairs(input) do
        local expected = (x > 0) and lambda or lambda * alpha * math.exp(x)
        luaunit.assertAlmostEquals(output.r[i], expected, 1e-3)
        luaunit.assertAlmostEquals(output.i[i], expected, 1e-3)
        luaunit.assertAlmostEquals(output.j[i], expected, 1e-3)
        luaunit.assertAlmostEquals(output.k[i], expected, 1e-3)
    end
end

function TestActivations:testSwish()
    local beta = 1.0
    local input = {-1, 0, 1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.swish(q_input, beta)
    local output = toTableQuat(q_output)
    for i, x in ipairs(input) do
        local expected = x * (1/(1+math.exp(-beta * x)))
        luaunit.assertAlmostEquals(output.r[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.i[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.j[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.k[i], expected, 1e-4)
    end
end

function TestActivations:testSwishDerivative()
    local beta = 1.0
    local input = {-1, 0, 1}
    local q_input = createQuaternion(input)
    local q_output = luann_af_q.activations.swish_derivative(q_input, beta)
    local output = toTableQuat(q_output)
    for i, x in ipairs(input) do
        local sig = 1/(1+math.exp(-beta*x))
        local expected = sig + beta * x * sig * (1 - sig)
        luaunit.assertAlmostEquals(output.r[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.i[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.j[i], expected, 1e-4)
        luaunit.assertAlmostEquals(output.k[i], expected, 1e-4)
    end
end

--------------------------------------------------------------------------------
-- Tests for Quaternion Weight Initialization Strategies
--------------------------------------------------------------------------------
TestWeightInits = {}

function TestWeightInits:testDefault()
    local dims = {4, 3}
    local W = luann_af_q.weightInits.default(3, 4, dims)
    local d0, d1 = af.af_get_dims(W.r)
    luaunit.assertEquals(d0, dims[1])
    luaunit.assertEquals(d1, dims[2])
    local d0_i, d1_i = af.af_get_dims(W.i)
    luaunit.assertEquals(d0_i, dims[1])
    luaunit.assertEquals(d1_i, dims[2])
end

function TestWeightInits:testXavier()
    local dims = {4, 3}
    local W = luann_af_q.weightInits.xavier(3, 4, dims)
    local d0, d1 = af.af_get_dims(W.r)
    luaunit.assertEquals(d0, dims[1])
    luaunit.assertEquals(d1, dims[2])
end

--------------------------------------------------------------------------------
-- Tests for the Quaternion Neural Network (Network module)
--------------------------------------------------------------------------------
TestNetwork = {}

function TestNetwork:testForward()
    local config = {
        layers = {3, 4, 2},
        activations = {"sigmoid", "softmax"},
        learningRate = 0.01,
        weightInitMethod = "xavier",
        dropoutRates = {0, 0}
    }
    local net = luann_af_q.Network:new(config)
    local input = {0.5, 0.2, -0.1}
    net:setInputSignals(createQuaternion(input))
    local output = net:forward()
    local outTable = toTableQuat(output)
    luaunit.assertEquals(#outTable.r, 2)
    if net.layers[#net.layers].activation == "softmax" then
        local sum = 0
        for i = 1, #outTable.r do
            sum = sum + outTable.r[i]
        end
        luaunit.assertAlmostEquals(sum, 1, 1e-4)
    end
end

function TestNetwork:testTrain()
    local config = {
        layers = {3, 4, 2},
        activations = {"sigmoid", "softmax"},
        learningRate = 0.01,
        weightInitMethod = "xavier",
        dropoutRates = {0, 0}
    }
    local net = luann_af_q.Network:new(config)
    local input = {0.5, 0.2, -0.1}
    net:setInputSignals(createQuaternion(input))
    local output1 = net:forward()
    local target = createQuaternion({0, 1})
    net:backpropagate(target)
    local output2 = net:forward()
    local table1 = toTableQuat(output1)
    local table2 = toTableQuat(output2)
    local diff = 0
    for i = 1, #table1.r do
        diff = diff + math.abs(table1.r[i] - table2.r[i])
    end
    luaunit.assertTrue(diff > 0, "Training did not update network outputs")
end

function TestNetwork:testGoodnessCalculation()
    local config = {
        layers = {3, 4, 2},
        activations = {"sigmoid", "sigmoid"},
        learningRate = 0.01,
        weightInitMethod = "default",
        dropoutRates = {0, 0}
    }
    local net = luann_af_q.Network:new(config)
    local input = {0.1, 0.2, 0.3}
    net:setInputSignals(createQuaternion(input))
    net:forward()
    local goodness = net:calculateGoodness(net.as[#net.as])
    luaunit.assertTrue(goodness >= 0, "Goodness should be non-negative")
end

--------------------------------------------------------------------------------
-- Tests for Network Extensions and Additional Methods
--------------------------------------------------------------------------------
TestNetworkExtensions = {}

function TestNetworkExtensions:testUpdateLearningRate()
    local config = {
        layers = {3, 4, 2},
        activations = {"sigmoid", "softmax"},
        learningRate = 0.01,
        weightInitMethod = "xavier",
        dropoutRates = {0, 0}
    }
    local net = luann_af_q.Network:new(config)
    net:updateLearningRate(0.05)
    luaunit.assertAlmostEquals(net.learningRate, 0.05, 1e-4)
end

function TestNetworkExtensions:testForwardForward()
    local config = {
        layers = {3, 4, 2},
        activations = {"sigmoid", "sigmoid"},
        learningRate = 0.01,
        weightInitMethod = "xavier",
        dropoutRates = {0, 0}
    }
    local net = luann_af_q.Network:new(config)
    net:setInputSignals(createQuaternion({0.1, 0.2, 0.3}))
    local goodness = net:forwardForward({"sigmoid", "sigmoid"}, nil, true)
    luaunit.assertEquals(type(goodness), "table")
    for _, g in pairs(goodness) do
        luaunit.assertTrue(g >= 0)
    end
end

function TestNetworkExtensions:testBackpropagateWithAttention()
    local config = {
        layers = {3, {numCells = 4, isAttention = true}, 2},
        activations = {"sigmoid", "softmax"},
        learningRate = 0.01,
        weightInitMethod = "xavier",
        dropoutRates = {0, 0}
    }
    local net = luann_af_q.Network:new(config)
    net:setInputSignals(createQuaternion({0.5, 0.2, -0.1}))
    local output1 = net:forward()
    local target = createQuaternion({0, 1})
    net:backpropagateWithAttention(target)
    local output2 = net:forward()
    local table1 = toTableQuat(output1)
    local table2 = toTableQuat(output2)
    local diff = 0
    for i = 1, #table1.r do
        diff = diff + math.abs(table1.r[i] - table2.r[i])
    end
    luaunit.assertTrue(diff > 0, "Backpropagation with attention did not update network outputs")
end

function TestNetworkExtensions:testGetSignals()
    local config = {
        layers = {3, 4, 2},
        activations = {"sigmoid", "softmax"},
        learningRate = 0.01,
        weightInitMethod = "xavier",
        dropoutRates = {0, 0}
    }
    local net = luann_af_q.Network:new(config)
    net:setInputSignals(createQuaternion({0.5, 0.2, -0.1}))
    net:forward()
    local signals = net:getSignals(2)
    luaunit.assertEquals(type(signals), "table")
    luaunit.assertEquals(#signals, 4)
end

--------------------------------------------------------------------------------
-- Tests for the Quaternion LSTM Cell
--------------------------------------------------------------------------------
TestLSTMCell = {}

function TestLSTMCell:testForward()
    local lstm = luann_af_q.LSTMCell:new(3, 2, "default")
    local input = {0.1, -0.2, 0.3}
    local q_input = createQuaternion(input)
    local output = lstm:forward(q_input)
    local outTable = toTableQuat(output)
    luaunit.assertEquals(#outTable.r, 2)
end

--------------------------------------------------------------------------------
-- Tests for the Quaternion Attention Module Extensions
--------------------------------------------------------------------------------
TestAttentionExtensions = {}

function TestAttentionExtensions:testMultiHead()
    -- Create a dummy quaternion query matrix with shape 2x4 (2 rows, 4 columns)
    local queryData = {1, 2, 3, 4, 5, 6, 7, 8}
    local query = createQuaternionFromData(queryData, 2, {2, 4})
    local key = query
    local value = query
    local output = luann_af_q.Attention.multiHead(query, key, value, 2, 0)
    local d0, d1, _, _ = af.af_get_dims(output.r)
    luaunit.assertEquals(d0, 2)
    luaunit.assertEquals(d1, 4)
end

function TestAttentionExtensions:testSplitAndConcatHeads()
    -- Create a dummy quaternion matrix of shape 4x8
    local data = {}
    for i = 1, 32 do data[i] = i end
    local matrix = createQuaternionFromData(data, 2, {4, 8})
    local headSize = 4
    local head1 = luann_af_q.Attention.splitHead(matrix, 1, headSize)
    local head2 = luann_af_q.Attention.splitHead(matrix, 2, headSize)
    local concat = luann_af_q.Attention.concatHeads({head1, head2}, 8)
    local d0, d1, _, _ = af.af_get_dims(concat.r)
    luaunit.assertEquals(d0, 4)
    luaunit.assertEquals(d1, 8)
end

--------------------------------------------------------------------------------
-- Run the tests
--------------------------------------------------------------------------------
os.exit(luaunit.LuaUnit.run())
