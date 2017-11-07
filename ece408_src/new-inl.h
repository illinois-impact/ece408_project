
#ifndef MXNET_OPERATOR_NEW_INL_H_
#define MXNET_OPERATOR_NEW_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../nn/im2col.h"

#include <chrono>

namespace mxnet
{
namespace op
{

namespace conv
{
enum ConvolutionOpInputs
{
  kData,
  kWeight
};
enum ConvolutionOpOutputs
{
  kOut
};
}

struct NewParam : public dmlc::Parameter<NewParam>
{
  TShape kernel;
  uint32_t num_filter;
  DMLC_DECLARE_PARAMETER(NewParam)
  {
    DMLC_DECLARE_FIELD(kernel).describe("convolution kernel size: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000).describe("convolution filter(channel) number");
  }
};

template<typename xpu, typename DType>
void forward(mshadow::Tensor<xpu, 4, DType> &y, const mshadow::Tensor<xpu, 4, DType> &x, const mshadow::Tensor<xpu, 4, DType> &w);

template <typename xpu, typename DType>
class NewOp : public Operator
{
public:
  explicit NewOp(NewParam p)
  {
    this->param_ = p;
  }


  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args)
  {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[conv::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 2U);         // two inputs with no bias
    CHECK_EQ(out_data.size(), 1U);        // create a single output
    CHECK_EQ(req[conv::kOut], kWriteTo);  // expect to be overwriting output
    CHECK_EQ(this->param_.kernel[0], this->param_.kernel[1]); // ECE408: square kernel
    
    const auto &x = in_data[conv::kData];
    const auto &xshape = x.shape_;
    const auto &w = in_data[conv::kWeight];
    const auto &wshape = w.shape_;
    const auto &y = out_data[conv::kOut];
    const auto &yshape = y.shape_;
    
    CHECK_EQ(wshape.ndim(), 4U); // num_filter , channel  y, x
    CHECK_EQ(wshape[0], this->param_.num_filter); // ECE408: support 1 group
    CHECK_EQ(wshape[1], xshape[1]);
    CHECK_EQ(wshape[2], wshape[3]); // square kernel
    CHECK_EQ(xshape.ndim(), 4U); // batch, num_filter, y, x
    CHECK_EQ(yshape.ndim(), 4U);



    CHECK_EQ(yshape[0], xshape[0]);
    CHECK_EQ(yshape[1], this->param_.num_filter);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    // Get output data as a 4D tensor
    Tensor<xpu, 4, DType> y_4d = y.get_with_shape<xpu, 4, DType>(
        Shape4(yshape[0], yshape[1], yshape[2], yshape[3]), s);

    // Get kernel data as a 4D tensor
    Tensor<xpu, 4, DType> w_4d = w.get_with_shape<xpu, 4, DType>(
        Shape4(wshape[0], wshape[1], wshape[2], wshape[3]), s);

    // Get input data as a 4D tensor
    Tensor<xpu, 4, DType> x_4d = x.get_with_shape<xpu, 4, DType>(
        Shape4(xshape[0], xshape[1], xshape[2], xshape[3]), s);


  auto start = std::chrono::high_resolution_clock::now();
  forward<xpu, DType>(y_4d, x_4d, w_4d);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  fprintf(stdout, "Op Time: %f\n", elapsed_seconds.count());
              
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args)
  {
    // See https://bitbucket.org/hwuligans/2017fa_ece408_project_solution
    // for an implementation of this if you need to generate model parameters.
    // That repo should not be released to students
    CHECK_EQ(0,1) << "Backward pass unimplemented for ECE408";
  }

protected:
  NewParam param_;
}; // class NewOp

template <typename xpu>
Operator *CreateOp(NewParam param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);

#if DMLC_USE_CXX11
class NewProp : public OperatorProperty
{
public:
  std::vector<std::string> ListArguments() const override
  {
    return {"data", "weight"};
  }

  void Init(const std::vector<std::pair<std::string, std::string>> &kwargs) override
  {
    using namespace mshadow;
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override
  {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override
  {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
    out_shape->resize(1, TShape());

    CHECK_EQ(param_.kernel.ndim(), 2U) << "ECE408: only support 2D kernels";
    CHECK_EQ(param_.kernel[0], param_.kernel[1]) << "ECE408: only support square kernels";
    const int K = param_.kernel[0];

    const TShape &dshp = (*in_shape)[conv::kData];
    CHECK_EQ(dshp.ndim(), 4U) << "Input data should be 4D in batch-num_filter-y-x";
    const int C = dshp[1];
    const int H = dshp[2];
    const int W = dshp[3];
    const int M = param_.num_filter;

    Shape<4> dshape = dshp.get<4>();
    Shape<4> wshape = Shape4(M, C, K, K);
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kWeight, wshape);

    Shape<4> oshape = Shape4(dshape[0], M, H - K + 1, W - K + 1);
    SHAPE_ASSIGN_CHECK(*out_shape, 0, oshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override
  {

    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i)
    {
      if ((*in_type)[i] == -1)
      {
        (*in_type)[i] = dtype;
      }
      else
      {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty *Copy() const override
  {
    auto ptr = new NewProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override
  {
    return "New";
  }


  Operator *CreateOperator(Context ctx) const override
  {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

private:
  NewParam param_;
};     // class NewProp
#endif // DMLC_USE_CXX11
} // namespace op
} // namespace mxnet
#endif // MXNET_OPERATOR_NEW_INL_H_
