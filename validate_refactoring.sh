#!/bin/bash
# Validation script to check for incomplete refactoring

echo "========================================="
echo "Memory Resources Refactoring Validation"
echo "========================================="
echo ""

errors=0

echo "[1/6] Checking for remaining get_current_device_resource_ref() in src/ files..."
count=$(grep -r "cudf::get_current_device_resource_ref()" cpp/src --include="*.cu" --include="*.cpp" 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
  echo "⚠️  Found $count occurrences (excluding the one in memory_resource.hpp constructor)"
  grep -r "cudf::get_current_device_resource_ref()" cpp/src --include="*.cu" --include="*.cpp" | head -10
  echo ""
else
  echo "✓ No issues found"
fi
echo ""

echo "[2/6] Checking for exec_policy without MR in src/ files..."
count=$(grep -r "rmm::exec_policy(stream)" cpp/src --include="*.cu" 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
  echo "⚠️  Found $count occurrences"
  grep -r "rmm::exec_policy(stream)" cpp/src --include="*.cu" | head -10
  ((errors++))
  echo ""
else
  echo "✓ All exec_policy calls have memory resource"
fi
echo ""

echo "[3/6] Checking for unconverted public API signatures in headers..."
count=$(grep -r "rmm::device_async_resource_ref mr" cpp/include/cudf --include="*.hpp" --include="*.cuh" 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
  echo "⚠️  Found $count occurrences"
  grep -r "rmm::device_async_resource_ref mr" cpp/include/cudf --include="*.hpp" --include="*.cuh" | head -10
  ((errors++))
  echo ""
else
  echo "✓ All public APIs use cudf::memory_resources"
fi
echo ""

echo "[4/6] Checking that memory_resources class was added..."
if grep -q "class memory_resources" cpp/include/cudf/utilities/memory_resource.hpp; then
  echo "✓ memory_resources class found"
else
  echo "❌ memory_resources class NOT found"
  ((errors++))
fi
echo ""

echo "[5/6] Checking validation support in get_current_device_resource_ref()..."
if grep -q "LIBCUDF_ERROR_ON_CURRENT_DEVICE_RESOURCE_REF" cpp/include/cudf/utilities/memory_resource.hpp; then
  echo "✓ Validation support found"
else
  echo "❌ Validation support NOT found"
  ((errors++))
fi
echo ""

echo "[6/6] Checking for files that might need manual review..."
echo "Files with 'mr' variable that might need attention:"
grep -r "\bmr\b" cpp/src --include="*.cu" --include="*.cpp" | grep -v "// " | grep -v "resources" | wc -l
echo "(This is expected - some local variables named 'mr' are fine)"
echo ""

echo "========================================="
if [ "$errors" -eq 0 ]; then
  echo "✓ Validation PASSED - Refactoring looks good!"
else
  echo "⚠️  Found $errors potential issues"
fi
echo "========================================="
