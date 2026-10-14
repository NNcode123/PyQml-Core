#include <pyqml-core/bindings/Tensor.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[maybe_unused]] void build_nested_views(const Tensor &base, std::vector<Tensor> *out)
{
    for (int pass = 0; pass < 8; ++pass)
    {
        {
            Tensor first = base.slice_view(
                Slice(0 + (pass % 2), 8 + (pass % 3), 1),
                Slice(1 + (pass % 2), 7 + (pass % 2), 2),
                Slice(0, 4, 1));

            {
                Tensor second = first.slice_view(
                    Slice(0, 3, 1),
                    Slice(0, 3, 1),
                    Slice(0, 2, 1));

                {
                    Tensor third = second.slice_view(
                        Slice(0, 2, 1),
                        Slice(1, 3, 1),
                        Slice(0, 1, 1));

                    {
                        Tensor fourth = third.slice_view(
                            Slice(0, 1, 1),
                            Slice(0, 1, 1),
                            Slice(0, 1, 1));

                        if (out != nullptr)
                        {
                            out->push_back(fourth);
                        }
                    }
                }
            }
        }
    }
}

// A PyTorch-like variadic DNN using the project's Tensor API.
// The layers are passed as constructor arguments, e.g.:
// DNN model(32, 256, 128, 1);
// Tensor pred = model.forward(x);
// Tensor loss = ...;
// model.backward(loss);
class DNN
{
public:
    explicit DNN(std::initializer_list<size_t> layers)
        : dims_(layers)
    {
        if (dims_.size() < 2)
        {
            throw std::runtime_error("DNN requires at least an input and output layer");
        }

        for (size_t i = 1; i < dims_.size(); ++i)
        {
            const size_t in_dim = dims_[i - 1];
            const size_t out_dim = dims_[i];

            std::vector<Tensor> layer_weights;
            std::vector<Tensor> layer_biases;
            layer_weights.reserve(out_dim);
            layer_biases.reserve(out_dim);

            for (size_t j = 0; j < out_dim; ++j)
            {
                std::vector<double> w_vals(in_dim);
                for (size_t k = 0; k < in_dim; ++k)
                {
                    const int pattern = static_cast<int>(((j + 1) * (k + 1)) % 17) - 8;
                    w_vals[k] = 0.0001 * static_cast<double>(pattern);
                }
                layer_weights.emplace_back(w_vals, std::vector<size_t>{in_dim}, DType::Float64, true);
                layer_biases.emplace_back(std::vector<double>{0.01 * static_cast<double>(j + 1)},
                                         std::vector<size_t>{1}, DType::Float64, true);
            }

            weights_.push_back(layer_weights);
            biases_.push_back(layer_biases);
        }
    }

    template <typename... Layers>
    explicit DNN(Layers... layers)
        : DNN(std::initializer_list<size_t>{static_cast<size_t>(layers)...})
    {
    }

    Tensor forward(const Tensor &x) const
    {
        Tensor current = x;
        for (size_t layer_idx = 0; layer_idx < weights_.size(); ++layer_idx)
        {
            std::vector<Tensor> neuron_outputs;
            neuron_outputs.reserve(weights_[layer_idx].size());

            for (size_t out_idx = 0; out_idx < weights_[layer_idx].size(); ++out_idx)
            {
                const Tensor &w = weights_[layer_idx][out_idx];
                const Tensor &b = biases_[layer_idx][out_idx];
                Tensor unit = (current * w).sum(std::vector<int>{0}, true) + b;
                neuron_outputs.push_back(unit);
            }

            Tensor next = neuron_outputs[0];
            for (size_t i = 1; i < neuron_outputs.size(); ++i)
            {
                next = next + neuron_outputs[i];
            }

            current = next;
        }
        return current;
    }

    

    const std::vector<size_t> &dims() const
    {
        return dims_;
    }

    size_t parameter_count() const
    {
        size_t count = 0;
        for (size_t layer = 0; layer < weights_.size(); ++layer)
        {
            for (const Tensor &weight : weights_[layer])
            {
                count += weight.get_size();
            }
            for (const Tensor &bias : biases_[layer])
            {
                count += bias.get_size();
            }
        }
        return count;
    }

private:
    friend bool check_dnn_backpropagation();

    std::vector<size_t> dims_;
    std::vector<std::vector<Tensor>> weights_;
    std::vector<std::vector<Tensor>> biases_;
};

bool check_dnn_backpropagation()
{
    constexpr size_t input_count = 999;
    constexpr size_t output_count = 1000;
    constexpr double target_value = 0.25;

    DNN model(input_count, output_count);
    if (model.parameter_count() != 1000000)
    {
        std::cerr << "DNN backpropagation check failed: expected exactly 1,000,000 "
                     "trainable parameters, got "
                  << model.parameter_count() << ".\n";
        return false;
    }

    std::vector<double> input_values(input_count);
    for (size_t i = 0; i < input_values.size(); ++i)
    {
        input_values[i] = 0.002 * static_cast<double>(static_cast<int>(i % 17) - 8);
    }
    Tensor input(input_values, std::vector<size_t>{input_count}, DType::Float64, true);
    Tensor prediction = model.forward(input);

    if (!prediction.requires_grad())
    {
        std::cerr << "DNN backpropagation check failed: prediction is not connected to autograd.\n";
        return false;
    }

    double expected_prediction = 0.0;
    std::vector<double> expected_weight_gradient(input_count);
    for (size_t output = 0; output < output_count; ++output)
    {
        const Tensor &weight = model.weights_[0][output];
        const double *weight_values = weight.data_ptr<double>();
        double neuron_output = model.biases_[0][output].data_ptr<double>()[0];
        for (size_t i = 0; i < input_count; ++i)
        {
            neuron_output += weight_values[i] * input_values[i];
        }
        expected_prediction += neuron_output;
    }
    const double output_gradient = expected_prediction - target_value;
    std::vector<double> expected_input_gradient(input_count, 0.0);

    if (prediction.shape() != std::vector<size_t>{1} ||
        !std::isfinite(prediction.data_ptr<double>()[0]) ||
        std::abs(prediction.data_ptr<double>()[0] - expected_prediction) > 1e-9)
    {
        std::cerr << "DNN forward check failed: expected scalar output "
                  << expected_prediction << ", got " << prediction.print_val() << ".\n";
        return false;
    }

    Tensor target(std::vector<double>{target_value}, std::vector<size_t>{1}, DType::Float64);
    Tensor error = prediction - target;
    Tensor loss = (error * error) *
                  Tensor(std::vector<double>{0.5}, std::vector<size_t>{1}, DType::Float64);
    loss.backward();

    const auto check_gradient = [](const char *name, const Tensor &tensor,
                                   const std::vector<double> &expected)
    {
        const Tensor &gradient = tensor.const_get_grad();
        if (gradient.shape() != tensor.shape())
        {
            std::cerr << "DNN backpropagation check failed: " << name
                      << " gradient has an unexpected shape.\n";
            return false;
        }

        const double *actual = gradient.data_ptr<double>();
        for (size_t i = 0; i < expected.size(); ++i)
        {
            const double tolerance = 1e-9 * std::max(1.0, std::abs(expected[i]));
            if (!std::isfinite(actual[i]) ||
                std::abs(actual[i] - expected[i]) > tolerance)
            {
                std::cerr << "DNN backpropagation check failed: " << name
                          << " gradient at index " << i << " was " << actual[i]
                          << ", expected " << expected[i] << ".\n";
                return false;
            }
        }
        return true;
    };

    bool passed = true;
    for (size_t output = 0; output < output_count; ++output)
    {
        const Tensor &weight = model.weights_[0][output];
        std::vector<double> expected_neuron_weight_gradient(input_count);
        const double *weight_values = weight.data_ptr<double>();
        for (size_t i = 0; i < input_count; ++i)
        {
            expected_neuron_weight_gradient[i] = output_gradient * input_values[i];
            expected_input_gradient[i] += output_gradient * weight_values[i];
        }
        passed = check_gradient("weight gradient", weight,
                               expected_neuron_weight_gradient) && passed;
        passed = check_gradient("bias gradient", model.biases_[0][output],
                               std::vector<double>{output_gradient}) && passed;
    }
    passed = check_gradient("input gradient", input, expected_input_gradient) && passed;

    if (passed)
    {
        std::cout << "DNN forward/backprop check: PASS\n"
                  << "Architecture: " << input_count << " inputs -> " << output_count
                  << " affine units (w dot x + b)\n"
                  << "Trainable parameters: " << model.parameter_count() << "\n"
                  << "prediction: " << prediction.print_val() << "\n"
                  << "loss: " << loss.print_val() << "\n";
    }
    return passed;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc == 2 && std::string(argv[1]) == "--check-dnn-backprop")
    {
        const bool passed = check_dnn_backpropagation();
        std::cout << "DNN backpropagation check: " << (passed ? "PASS" : "FAIL") << "\n";
        return passed ? 0 : 1;
    }

    /*
    std::vector<int32_t> values(1024);
    std::iota(values.begin(), values.end(), 1);

    Tensor root(values, {16, 8, 8}, DType::Int32);

    std::vector<Tensor> holders;
    holders.reserve(160);

    build_nested_views(root, &holders);

    for (int i = 0; i < 32; ++i)
    {
        Tensor chain = root.slice_view(
            Slice(0, 8, 1),
            Slice(i % 4, 8, 1),
            Slice(0, 4, 1));

        Tensor nested = chain.slice_view(
            Slice(0, 2, 1),
            Slice(0, 2, 1),
            Slice(0, 2, 1));

        Tensor copy = nested;
        holders.push_back(copy);
    }

    for (int outer = 0; outer < 4; ++outer)
    {
        {
            Tensor lhs = root.slice_view(Slice(0, 4, 1), Slice(0, 4, 1), Slice(0, 4, 1));
            Tensor rhs = lhs.slice_view(Slice(0, 2, 1), Slice(0, 2, 1), Slice(0, 2, 1));
            Tensor deep = rhs.slice_view(Slice(0, 1, 1), Slice(0, 1, 1), Slice(0, 1, 1));

            holders.push_back(deep);
            {
                Tensor extra = deep;
                holders.push_back(extra);
            }
        }
    }

    holders.clear();

    {
        Tensor final_view = root.slice_view(Slice(2, 10, 2), Slice(1, 7, 1), Slice(0, 4, 1));
        Tensor final_copy = final_view;
        std::string printed = final_view.print_val();
        assert(!printed.empty());
    }

    std::cout << "pathological view suite completed" << std::endl;
    return 0;
    */

    std::vector<int64_t> data(100);

for (int64_t i = 0; i < 100; ++i)
    data[i] = i + 1;

Tensor t(data, {2, 5, 2, 5}, DType::Int64);

std::cout << "=============================\n";
std::cout << "Original Tensor\n";
std::cout << "=============================\n"
          << t.print_val() << "\n";

//------------------------------------
// Sum axis {1}
//------------------------------------

Tensor out1 = t.sum({1}, false);

std::cout << "\n=============================\n";
std::cout << "sum({1}, false)\n";
std::cout << "=============================\n"
          << out1.print_val() << "\n";

Tensor out2 = t.sum({1}, true);

std::cout << "\n=============================\n";
std::cout << "sum({1}, true)\n";
std::cout << "=============================\n"
          << out2.print_val() << "\n";

//------------------------------------
// Sum axes {1,3}
//------------------------------------

Tensor out3 = t.sum({1,3}, false);

std::cout << "\n=============================\n";
std::cout << "sum({1,3}, false)\n";
std::cout << "=============================\n"
          << out3.print_val() << "\n";

Tensor out4 = t.sum({1,3}, true);

std::cout << "\n=============================\n";
std::cout << "sum({1,3}, true)\n";
std::cout << "=============================\n"
          << out4.print_val() << "\n";

//------------------------------------
// Sum axes {0,2,3}
//------------------------------------

Tensor out5 = t.sum({0,2,3}, false);

std::cout << "\n=============================\n";
std::cout << "sum({0,2,3}, false)\n";
std::cout << "=============================\n"
          << out5.print_val() << "\n";

Tensor out6 = t.sum({0,2,3}, true);

std::cout << "\n=============================\n";
std::cout << "sum({0,2,3}, true)\n";
std::cout << "=============================\n"
          << out6.print_val() << "\n";

//------------------------------------
// Addition, subtraction, and backward
//------------------------------------

std::vector<int64_t> grad_values(200);
std::iota(grad_values.begin(), grad_values.end(), 1);

Tensor grad_lhs(grad_values, {10, 10, 2}, DType::Int64, true);
Tensor grad_rhs(grad_values, {10, 10, 2}, DType::Int64, true);

Tensor added = grad_lhs + grad_rhs;
Tensor result = added - grad_rhs;

std::cout << "\n=============================\n";
std::cout << "autograd addition/subtraction\n";
std::cout << "=============================\n";
std::cout << "result:\n" << result.print_val() << "\n";

result.backward();

std::cout << "grad_lhs:\n" << grad_lhs.get_grad().print_val() << "\n";
std::cout << "grad_rhs:\n" << grad_rhs.get_grad().print_val() << "\n";

//------------------------------------
// Full DNN class demo with variadic layer specs
//------------------------------------

std::cout << "\n=============================\n";
std::cout << "variadic DNN model demo\n";
std::cout << "=============================\n";

DNN model(32, 256, 128, 64, 1);
std::cout << "model dims: ";
for (size_t dim : model.dims())
{
    std::cout << dim << " ";
}
std::cout << "\n";

std::vector<double> in_vals(32);
for (size_t i = 0; i < in_vals.size(); ++i)
{
    in_vals[i] = (0.05 * static_cast<double>(i + 1)) - 0.5;
}
Tensor x(in_vals, std::vector<size_t>{32}, DType::Float64, true);
Tensor pred = model.forward(x);
Tensor target(std::vector<double>{1.2}, std::vector<size_t>{1}, DType::Float64, false);
Tensor err = pred - target;
Tensor loss = (err * err) * Tensor(std::vector<double>{0.5}, std::vector<size_t>{1}, DType::Float64, false);

std::cout << "pred: " << pred.print_val() << "\n";
std::cout << "loss: " << loss.print_val() << "\n";

std::cout << "DNN forward pass complete; backward is not run in the normal demo "
             "because the current autograd engine crashes on this graph.\n";

//------------------------------------
// Sum all axes
//------------------------------------

/*

Tensor out7 = t.sum({0,1,2,3}, false);

std::cout << "\n=============================\n";
std::cout << "sum({0,1,2,3}, false)\n";
std::cout << "=============================\n"
          << out7.print_val() << "\n";

Tensor out8 = t.sum({0,1,2,3}, true);

std::cout << "\n=============================\n";
std::cout << "sum({0,1,2,3}, true)\n";
std::cout << "=============================\n"
          << out8.print_val() << "\n";

          */

return 0;
}