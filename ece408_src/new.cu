
#include "./new-inl.h"
#include "./new-forward.cuh"
#include <vector>


namespace mxnet {
namespace op {


template<>
Operator* CreateOp<gpu>(NewParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new NewOp<gpu, DType>(param);
  })
  return op;

}

}  // namespace op
}  // namespace mxnet

