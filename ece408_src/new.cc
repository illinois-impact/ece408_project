
#include "./new-inl.h"
#include "./new-forward.h"


namespace mxnet {
namespace op {


  DMLC_REGISTER_PARAMETER(NewParam);

template<>
Operator* CreateOp<cpu>(NewParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new NewOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *NewProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(New, NewProp)
.describe(R"code()code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the NewOp.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_arguments(NewParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
