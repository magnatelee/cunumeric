/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "scalar_binary_op.h"
#include "binary_op_util.h"
#include "core.h"
#include "dispatch.h"
#include "scalar.h"

namespace legate {
namespace numpy {

using namespace Legion;

template <BinaryOpCode OP_CODE>
struct BinaryOpImpl {
  template <LegateTypeCode CODE, std::enable_if_t<BinaryOp<OP_CODE, CODE>::valid> * = nullptr>
  UntypedScalar operator()(UntypedScalar &in1, UntypedScalar &in2)
  {
    using OP  = BinaryOp<OP_CODE, CODE>;
    using ARG = legate_type_of<CODE>;
    using RES = std::result_of_t<OP(ARG, ARG)>;

    OP func{};
    return UntypedScalar(func(in1.value<ARG>(), in2.value<ARG>()));
  }

  template <LegateTypeCode CODE, std::enable_if_t<!BinaryOp<OP_CODE, CODE>::valid> * = nullptr>
  UntypedScalar operator()(UntypedScalar &in1, UntypedScalar &in2)
  {
    assert(false);
    return UntypedScalar();
  }
};

struct BinaryOpDispatch {
  template <BinaryOpCode OP_CODE>
  UntypedScalar operator()(UntypedScalar &in1, UntypedScalar &in2)
  {
    return type_dispatch(in1.code(), BinaryOpImpl<OP_CODE>{}, in1, in2);
  }
};

/*static*/ UntypedScalar ScalarBinaryOpTask::cpu_variant(const Task *task,
                                                         const std::vector<PhysicalRegion> &regions,
                                                         Context context,
                                                         Runtime *runtime)
{
  Deserializer ctx(task, regions);

  BinaryOpCode op_code;
  UntypedScalar in1;
  UntypedScalar in2;

  deserialize(ctx, op_code);
  deserialize(ctx, in1);
  deserialize(ctx, in2);

  return op_dispatch(op_code, BinaryOpDispatch{}, in1, in2);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  ScalarBinaryOpTask::register_variants_with_return<UntypedScalar, UntypedScalar>();
}
}  // namespace

}  // namespace numpy
}  // namespace legate
