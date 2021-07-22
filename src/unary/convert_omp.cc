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

#include "unary/convert.h"
#include "unary/convert_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <LegateTypeCode DST_TYPE, LegateTypeCode SRC_TYPE, int DIM>
struct ConvertImplBody<VariantKind::OMP, DST_TYPE, SRC_TYPE, DIM> {
  using OP  = ConvertOp<DST_TYPE, SRC_TYPE>;
  using SRC = legate_type_of<SRC_TYPE>;
  using DST = legate_type_of<DST_TYPE>;

  void operator()(OP func,
                  AccessorWO<DST, DIM> out,
                  AccessorRO<SRC, DIM> in,
                  const Pitches<DIM - 1> &pitches,
                  const Rect<DIM> &rect,
                  bool dense) const
  {
    const size_t volume = rect.volume();
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
#pragma omp parallel for schedule(static)
      for (size_t idx = 0; idx < volume; ++idx) outptr[idx] = func(inptr[idx]);
    } else {
      OMPLoop<DIM>::unary_loop(func, out, in, rect);
    }
  }
};

/*static*/ void ConvertTask::omp_variant(TaskContext &context)
{
  convert_template<VariantKind::OMP>(context);
}

}  // namespace numpy
}  // namespace legate
