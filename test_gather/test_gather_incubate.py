import numpy as np
import paddle
# import torch
import unittest
from paddle.fluid.layers.utils import map_structure
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

class TestGatherIncubateCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        # x_torch, index_torch, dout_torch = self.gen_torch_inputs_and_dout()
        # out_torch, out_grads_torch = self.cal_torch_res(x_torch, index_torch, dout_torch)
        # del x_torch 
        # del index_torch 
        # del dout_torch 
        # self.out_torch = out_torch.cpu().detach().numpy()
        # self.out_grads_torch = map_structure(
        #                         lambda x: x.cpu().numpy(),
        #                         out_grads_torch,
        #                     )
        # del out_torch, out_grads_torch
        # torch.cuda.empty_cache()

    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case1_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp32.npz"
    
    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        np_inputs_array = np.load(self.np_input_dir)
        # get np array from npz file
        self.np_x = np_inputs_array["x"]
        self.np_index = np_inputs_array["index"]
        self.np_dout = np_inputs_array["dout"]
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")
    
    # def gen_torch_inputs_and_dout(self):
    #     x_torch = torch.tensor(
    #         self.np_x,
    #         device='cuda',
    #         dtype=convert_dtype_to_torch_type(self.dtype)
    #         if self.dtype != 'bfloat16'
    #         else torch.float32,
    #         requires_grad=True,
    #     )
    #     index_torch = torch.tensor(
    #         self.np_index,
    #         device='cuda',
    #         dtype=torch.int32,
    #         # requires_grad=True,
    #     )
    #     dout_torch = torch.tensor(
    #         self.np_dout,
    #         device='cuda',
    #         dtype=convert_dtype_to_torch_type(self.dtype)
    #         if self.dtype != 'bfloat16'
    #         else torch.float32,
    #         requires_grad=True,
    #     )
    #     return x_torch, index_torch, dout_torch
    
    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        index_eager = paddle.to_tensor(
            self.np_index,
            dtype="int32",
            place="gpu",
        )
        index_eager.stop_gradient = True
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, index_eager, dout_eager

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        index_static = paddle.static.data(
            'index',
            shape=self.np_index.shape,
            dtype="int32",
        )
        index_static.stop_gradient = True
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, index_static, dout_static

    # def cal_torch_res(self, x, index, dout):
    #     x_t = x
    #     index_t = index
    #     dout_t = dout
    #     if self.dtype == "bfloat16":
    #         x_t = x.to(dtype=torch.bfloat16)
    #         dout_t = dout.to(dtype=torch.bfloat16)
    #     index_t = index.to(dtype=torch.int64)
    #     out = torch.gather(x_t, index_t)
    #     out_grads = torch.autograd.grad([out], [x], grad_outputs=[dout_t])
    #     if self.dtype == "bfloat16":
    #         out = out.to(dtype=torch.float32)
    #     return out, out_grads

    def cal_eager_res(self, x, index, dout):
        x_t = x
        index_t = index
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.fluid.layers.gather(x_t, index_t)
        out_grads = paddle.grad(
            [out], [x], grad_outputs=[dout_t], retain_graph=True
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def cal_static_res(self, x, index, dout):
        x_t = x
        index_t = index
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.fluid.layers.gather(x_t, index_t)
        out_grads = paddle.static.gradients(
            [out], [x], target_gradients=[dout_t]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):
        # get develop eager res
        develop_res_array = np.load(self.save_eager_res_path)
        out_eager_develop = develop_res_array["out_eager"]
        out_eager_grad_0_develop = develop_res_array["out_grads_eager_0"]
        out_eager_grads_develop = [out_eager_grad_0_develop]

        # calculate incubate eager res
        x_eager, index_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(x_eager, index_eager, dout_eager)
        del x_eager
        del index_eager 
        del dout_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        out_grads_eager_np = map_structure(
                                lambda x: x.numpy(),
                                out_grads_eager,
                            )
        del out_eager
        del out_grads_eager
        paddle.device.cuda.empty_cache()
        # compare incubate eager res with develop eager res
        np.testing.assert_equal(
            out_eager_np,
            out_eager_develop,
            err_msg=(
                'Incubate: compare gather incubate eager forward res with develop eager forward res failed in %s dtype'
            )
            % self.dtype,
        )
        for idx in range(len(out_grads_eager_np)):
            np.testing.assert_equal(
                out_grads_eager_np[idx],
                out_eager_grads_develop[idx],
            err_msg=(
                'Incubate: compare gather incubate eager grad res with develop eager grad res failed in %s dtype'
            )
                % self.dtype,
            )
    
    def test_static_accuracy(self):
        # get develop static res
        develop_res_array = np.load(self.save_static_res_path)
        out_static_develop = develop_res_array["out_static"]
        out_grads_static_0_develop = develop_res_array["out_grads_static_0"]
        out_grads_static_develop = [out_grads_static_0_develop]

        # calculate incubate static res
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, index_static, dout_static = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    index_static,
                    dout_static,
                )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "index": self.np_index, "dout": self.np_dout},
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]
        
        # compare incubate static res with develop static res
        np.testing.assert_equal(
            out_static,
            out_static_develop,
            err_msg=(
                'Incubate: compare gather incubate static forward res with develop static forward res failed in %s dtype'
            )
            % self.dtype,
        )
        for idx in range(len(out_grads_static)):
            np.testing.assert_equal(
                out_grads_static[idx],
                out_grads_static_develop[idx],
            err_msg=(
                'Incubate: compare gather incubate static grad res with develop static grad res failed in %s dtype'
            )
                % self.dtype,
            )

    def test_eager_stability(self):
        x_eager, index_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(x_eager, index_eager, dout_eager)
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = map_structure(
                                lambda x: x.numpy(),
                                out_grads_eager_baseline,
                            )
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(50):
            out_eager, out_grads_eager = self.cal_eager_res(x_eager, index_eager, dout_eager)
            out_eager = out_eager.numpy()
            out_grads_eager = map_structure(
                                    lambda x: x.numpy(),
                                    out_grads_eager,
                                )
            np.testing.assert_equal(
                out_eager,
                out_eager_baseline_np,
                err_msg=(
                    'Incubate: paddle.fluid.layers.gather eager forward is unstable in %s dtype'
                )
                % self.dtype,
            )
            for idx in range(len(out_grads_eager)):
                np.testing.assert_equal(
                    out_grads_eager[idx],
                    out_grads_eager_baseline_np[idx],
                    err_msg=(
                        'Incubate: paddle.fluid.layers.gather eager grad is unstable in %s dtype'
                    )
                    % self.dtype,
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, index_static, dout_static = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    index_static,
                    dout_static,
                )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "index": self.np_index, "dout": self.np_dout},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "index": self.np_index, "dout": self.np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0], out[1:]
                np.testing.assert_equal(
                    out_static,
                    out_static_baseline,
                    err_msg=(
                        'Incubate: paddle.fluid.layers.gather static forward is unstable in %s dtype'
                    )
                    % self.dtype,
                )
                for idx in range(len(out_grads_static)):
                    np.testing.assert_equal(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        err_msg=(
                            'Incubate: paddle.fluid.layers.gather static grad is unstable in %s dtype'
                        )
                        % self.dtype,
                    )


class TestGatherIncubateCase1_FP16(TestGatherIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp16.npz"

class TestGatherIncubateCase1_BFP16(TestGatherIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case1_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_bfp16.npz"

if __name__ == '__main__':
    unittest.main()
