--------------------------------------------------------------------------------
-- test_luann_af.lua
-- Comprehensive unit tests for the luann_af neural network library using luaunit
--------------------------------------------------------------------------------

local luaunit = require("luaunit")
local luann_af = require("luann_af")
local af = require("luajit_af")

-- Helper: Convert an ArrayFire array to a Lua table.
local function toTable(afArray)
    local data_ptr, numElems = af.af_get_data_ptr(afArray)
    local t = {}
    for i = 0, numElems - 1 do
        t[#t + 1] = data_ptr[i]
    end
    return t
end

--------------------------------------------------------------------------------
-- Tests for Activation Functions
--------------------------------------------------------------------------------
TestActivations = {}

function TestActivations:testSigmoid()
    local input = {0, 1, -1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.sigmoid(af_input)
    local output = toTable(af_output)
    local expected = {0.5, 1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1))}
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-4)
    end
end

function TestActivations:testSigmoidDerivative()
    local input = {0, 1, -1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.sigmoid_derivative(af_input)
    local output = toTable(af_output)
    local expected = {}
    for i, x in ipairs(input) do
        local s = 1 / (1 + math.exp(-x))
        expected[i] = s * (1 - s)
    end
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-4)
    end
end

function TestActivations:testTanh()
    local input = {0, 1, -1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.tanh(af_input)
    local output = toTable(af_output)
    local expected = {math.tanh(0), math.tanh(1), math.tanh(-1)}
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-4)
    end
end

function TestActivations:testSoftmax()
    local input = {1, 2, 3}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.softmax(af_input)
    local output = toTable(af_output)
    local sum = 0
    for i = 1, #output do
        sum = sum + output[i]
    end
    luaunit.assertAlmostEquals(sum, 1, 1e-4)
end

function TestActivations:testRelu()
    local input = {-1, 0, 1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.relu(af_input)
    local output = toTable(af_output)
    local expected = {math.max(0, -1), math.max(0, 0), math.max(0, 1)}
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-4)
    end
end

function TestActivations:testReluDerivative()
    local input = {-1, 0, 1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.relu_derivative(af_input)
    local output = toTable(af_output)
    local expected = {0, 0, 1}
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-4)
    end
end

function TestActivations:testLeakyRelu()
    local alpha = 0.1
    local input = {-1, 0, 1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.leakyRelu(af_input, alpha)
    local output = toTable(af_output)
    local expected = {}
    for i, x in ipairs(input) do
        expected[i] = (x > 0) and x or alpha * x
    end
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-4)
    end
end

function TestActivations:testLeakyReluDerivative()
    local alpha = 0.1
    local input = {-1, 0, 1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.leakyRelu_derivative(af_input, alpha)
    local output = toTable(af_output)
    local expected = {}
    for i, x in ipairs(input) do
        expected[i] = (x > 0) and 1 or alpha
    end
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-4)
    end
end

function TestActivations:testElu()
    local alpha = 1.0
    local input = {-1, 0, 1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.elu(af_input, alpha)
    local output = toTable(af_output)
    local expected = {}
    for i, x in ipairs(input) do
        expected[i] = (x > 0) and x or alpha * (math.exp(x) - 1)
    end
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-4)
    end
end

function TestActivations:testEluDerivative()
    local alpha = 1.0
    local input = {-1, 0, 1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.elu_derivative(af_input, alpha)
    local output = toTable(af_output)
    local expected = {}
    for i, x in ipairs(input) do
        expected[i] = (x > 0) and 1 or alpha * math.exp(x)
    end
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-4)
    end
end

function TestActivations:testSelu()
    local lambda = 1.0507
    local alpha = 1.67326
    local input = {-1, 0, 1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.selu(af_input, lambda, alpha)
    local output = toTable(af_output)
    local expected = {}
    for i, x in ipairs(input) do
        expected[i] = (x > 0) and lambda * x or lambda * alpha * (math.exp(x) - 1)
    end
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-3)
    end
end

function TestActivations:testSeluDerivative()
    local lambda = 1.0507
    local alpha = 1.67326
    local input = {-1, 0, 1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.selu_derivative(af_input, lambda, alpha)
    local output = toTable(af_output)
    local expected = {}
    for i, x in ipairs(input) do
        expected[i] = (x > 0) and lambda or lambda * alpha * math.exp(x)
    end
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-3)
    end
end

function TestActivations:testSwish()
    local beta = 1.0
    local input = {-1, 0, 1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.swish(af_input, beta)
    local output = toTable(af_output)
    local expected = {}
    for i, x in ipairs(input) do
        expected[i] = x * (1 / (1 + math.exp(-beta * x)))
    end
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-4)
    end
end

function TestActivations:testSwishDerivative()
    local beta = 1.0
    local input = {-1, 0, 1}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local af_output = luann_af.activations.swish_derivative(af_input, beta)
    local output = toTable(af_output)
    local expected = {}
    for i, x in ipairs(input) do
        local sig = 1 / (1 + math.exp(-beta * x))
        expected[i] = sig + beta * x * sig * (1 - sig)
    end
    for i = 1, #expected do
        luaunit.assertAlmostEquals(output[i], expected[i], 1e-4)
    end
end

--------------------------------------------------------------------------------
-- Tests for Weight Initialization Strategies
--------------------------------------------------------------------------------
TestWeightInits = {}

function TestWeightInits:testDefault()
    local dims = {4, 3}
    local W = luann_af.weightInits.default(3, 4, dims)
    local d0, d1 = af.af_get_dims(W)
    luaunit.assertEquals(d0, dims[1])
    luaunit.assertEquals(d1, dims[2])
end

function TestWeightInits:testXavier()
    local dims = {4, 3}
    local W = luann_af.weightInits.xavier(3, 4, dims)
    local d0, d1 = af.af_get_dims(W)
    luaunit.assertEquals(d0, dims[1])
    luaunit.assertEquals(d1, dims[2])
end

--------------------------------------------------------------------------------
-- Tests for the Neural Network (Network module)
--------------------------------------------------------------------------------
TestNetwork = {}

function TestNetwork:testForward()
    local config = {
        layers = {3, 4, 2},               -- Input layer of size 3, hidden layer 4, output layer 2
        activations = {"sigmoid", "softmax"},
        learningRate = 0.01,
        weightInitMethod = "xavier",
        dropoutRates = {0, 0}
    }
    local net = luann_af.Network:new(config)
    local input = {0.5, 0.2, -0.1}
    net:setInputSignals(input)
    local output = net:forward()
    local outTable = toTable(output)
    luaunit.assertEquals(#outTable, 2)
    -- For softmax output, check that the probabilities sum to 1.
    if net.layers[#net.layers].activation == "softmax" then
        local sum = 0
        for i = 1, #outTable do
            sum = sum + outTable[i]
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
    local net = luann_af.Network:new(config)
    local input = {0.5, 0.2, -0.1}
    net:setInputSignals(input)
    local output1 = net:forward()
    local target = {0, 1}  -- target output vector
    net:backpropagate(target)
    local output2 = net:forward()
    local table1 = toTable(output1)
    local table2 = toTable(output2)
    local diff = 0
    for i = 1, #table1 do
        diff = diff + math.abs(table1[i] - table2[i])
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
    local net = luann_af.Network:new(config)
    local input = {0.1, 0.2, 0.3}
    net:setInputSignals(input)
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
    local net = luann_af.Network:new(config)
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
    local net = luann_af.Network:new(config)
    net:setInputSignals({0.1, 0.2, 0.3})
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
    local net = luann_af.Network:new(config)
    net:setInputSignals({0.5, 0.2, -0.1})
    local output1 = net:forward()
    local target = {0, 1}
    net:backpropagateWithAttention(target)
    local output2 = net:forward()
    local table1 = toTable(output1)
    local table2 = toTable(output2)
    local diff = 0
    for i = 1, #table1 do
        diff = diff + math.abs(table1[i] - table2[i])
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
    local net = luann_af.Network:new(config)
    net:setInputSignals({0.5, 0.2, -0.1})
    net:forward()
    local signals = net:getSignals(2)
    luaunit.assertEquals(type(signals), "table")
    luaunit.assertEquals(#signals, 4)
end

--------------------------------------------------------------------------------
-- Tests for the LSTM Cell
--------------------------------------------------------------------------------
TestLSTMCell = {}

function TestLSTMCell:testForward()
    local lstm = luann_af.LSTMCell:new(3, 2, "default")
    local input = {0.1, -0.2, 0.3}
    local af_input = af.af_create_array(input, 1, {#input}, af.f32)
    local output = lstm:forward(af_input)
    local outTable = toTable(output)
    luaunit.assertEquals(#outTable, 2)
end

--------------------------------------------------------------------------------
-- Tests for the Attention Module Extensions
--------------------------------------------------------------------------------
TestAttentionExtensions = {}

function TestAttentionExtensions:testMultiHead()
    -- Create a dummy query matrix with shape 2x4 (2 rows, 4 columns)
    local queryData = {1, 2, 3, 4, 5, 6, 7, 8}
    local query = af.af_create_array(queryData, 2, {2, 4}, af.f32)
    local key = query
    local value = query
    local output = luann_af.Attention.multiHead(query, key, value, 2, 0)
    local dims = {af.af_get_dims(output)}
    -- Expect the output to have the same shape as the input query.
    luaunit.assertEquals(dims[1], 2)
    luaunit.assertEquals(dims[2], 4)
end

function TestAttentionExtensions:testSplitAndConcatHeads()
    -- Create a dummy matrix of shape 4x8 (4 rows, 8 columns)
    local data = {}
    for i = 1, 32 do data[i] = i end
    local matrix = af.af_create_array(data, 2, {4, 8}, af.f32)
    local headSize = 4
    local head1 = luann_af.Attention.splitHead(matrix, 1, headSize)
    local head2 = luann_af.Attention.splitHead(matrix, 2, headSize)
    local concat = luann_af.Attention.concatHeads({head1, head2}, 8)
    local dims = {af.af_get_dims(concat)}
    luaunit.assertEquals(dims[1], 4)
    luaunit.assertEquals(dims[2], 8)
end

--------------------------------------------------------------------------------
-- Run the tests
--------------------------------------------------------------------------------
os.exit(luaunit.LuaUnit.run())
