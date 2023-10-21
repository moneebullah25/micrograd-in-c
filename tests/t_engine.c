#include <criterion/criterion.h>
#include "../includes/c_engine.h"
#include <stdio.h>

Test(ValueAdd, ValidAddition)
{
    Value *v1 = ValueNew(3.0);
    Value *v2 = ValueNew(4.0);

    Value *result = ValueAdd(v1, v2);

    cr_assert_not_null(result, "ValueAdd should return a non-NULL pointer");
    cr_assert_eq(result->data, 7.0, "ValueAdd should correctly add two values");
    cr_assert_eq(result->num_prev, 2, "ValueAdd should set the correct number of previous values");

    ValueFree(v1);
    ValueFree(v2);
    ValueFree(result);
}

Test(ValueMultiply, ValidMultiplication)
{
    Value *v1 = ValueNew(2.0);
    Value *v2 = ValueNew(3.0);

    Value *result = ValueMultiply(v1, v2);

    cr_assert_not_null(result, "ValueMultiply should return a non-NULL pointer");
    cr_assert_eq(result->data, 6.0, "ValueMultiply should correctly multiply two values");
    cr_assert_eq(result->num_prev, 2, "ValueMultiply should set the correct number of previous values");

    ValueFree(v1);
    ValueFree(v2);
    ValueFree(result);
}

Test(ValuePower, ValidPower)
{
    Value *v1 = ValueNew(2.0);
    Value *exponent = ValueNew(3.0);

    Value *result = ValuePower(v1, exponent);

    cr_assert_not_null(result, "ValuePower should return a non-NULL pointer");
    cr_assert_eq(result->data, 8.0, "ValuePower should correctly raise a value to an exponent");
    cr_assert_eq(result->num_prev, 2, "ValuePower should set the correct number of previous values");

    ValueFree(v1);
    ValueFree(exponent);
    ValueFree(result);
}

Test(ValueBackward, ValidBackward)
{
    Value *v1 = ValueNew(2.0);
    Value *v2 = ValueNew(3.0);

    Value *v3 = ValueMultiply(v1, v2);
    
    Value *v4 = ValuePower(v3, ValueNew(2.0));
    
    v4->grad = 1.0;
    ValueBackward(v4);
    
    cr_assert_eq(v1->grad, 36.0, "Gradient for v1 should be calculated correctly");
    cr_assert_eq(v2->grad, 24.0, "Gradient for v2 should be calculated correctly");
    
    ValueFree(v1);
    ValueFree(v2);
    ValueFree(v3);
    ValueFree(v4);
}

Test(ValueNegate, ValidNegation)
{
    Value *v1 = ValueNew(5.0);

    Value *result = ValueNegate(v1);

    cr_assert_not_null(result, "ValueNegate should return a non-NULL pointer");
    cr_assert_eq(result->data, -5.0, "ValueNegate should correctly negate a value");
    cr_assert_eq(result->num_prev, 1, "ValueNegate should set the correct number of previous values");

    ValueFree(v1);
    ValueFree(result);
}

Test(ValueSubtract, ValidSubtraction)
{
    Value *v1 = ValueNew(8.0);
    Value *v2 = ValueNew(3.0);

    Value *result = ValueSubtract(v1, v2);

    cr_assert_not_null(result, "ValueSubtract should return a non-NULL pointer");
    cr_assert_eq(result->data, 5.0, "ValueSubtract should correctly subtract two values");
    cr_assert_eq(result->num_prev, 2, "ValueSubtract should set the correct number of previous values");

    ValueFree(v1);
    ValueFree(v2);
    ValueFree(result);
}
