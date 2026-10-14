#pragma once




class Tensor;

struct grad_helper{

    static void attach_grad(Tensor& t) noexcept;

    static void requires_grad_(Tensor& t) noexcept;

    static void retain_grad_(Tensor& t) noexcept;

    //static void 



};
