/*
  This file is part of the ARG-Needle genealogical inference and
  analysis software suite.
  Copyright (C) 2023 ARG-Needle Developers.

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ARG_NEEDLE_LIB_GENOTYPE_MAPPING_HPP
#define ARG_NEEDLE_LIB_GENOTYPE_MAPPING_HPP

#include "arg.hpp"
#include "arg_edge.hpp"
#include "types.hpp"

#include <utility>
#include <vector>

namespace arg_utils
{

void map_genotype_to_ARG(ARG& arg, const std::vector<int>& genotype, int site_id);

std::pair<bool, std::vector<ARGEdge*>> map_genotype_to_ARG_relate(
    ARG& arg, const std::vector<int>& genotype, arg_real_t pos, double maf_threshold);

} // namespace arg_utils

#endif // ARG_NEEDLE_LIB_GENOTYPE_MAPPING_HPP
