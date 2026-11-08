#pragma once

#include "fixtures.h"

namespace ninfer::test::gdn_control {
int validate(Profile profile, bool normalize_input, DeviceExecutionView execution);
int validate_capacity(Profile profile, bool normalize_input);
} // namespace ninfer::test::gdn_control
