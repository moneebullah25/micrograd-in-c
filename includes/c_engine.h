#ifndef _C_ENGINE_HEADER_
#define _C_ENGINE_HEADER_

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#ifndef NULL
#define NULL 0
#endif

#define bool _Bool
#define false 0
#define true 1

#define EULER_CONST 2.71828182845904523536

    typedef struct Value
    {
        double data;
        double grad;
        void (*backward)(struct Value *);
        struct Value *_prev[2];
        int num_prev;
        int index;
    } Value;

    Value *ValueNew(double data);
    Value *ValueAdd(const Value *v1, const Value *v2);
    void ValueAddBackward(Value *v1);
    Value *ValueMultiply(const Value *v1, const Value *v2);
    void ValueMultiplyBackward(Value *v1);
    Value *ValuePower(const Value *v1, const Value *exponent);
    void ValuePowerBackward(Value *v1);
    Value *ValueReLU(const Value *v1);
    void ValueReLUBackward(Value *v1);
    Value *ValueTanh(const Value *v1);
    void ValueTanhBackward(Value *v1);
    void ValueBackward(Value *v1);
    Value *ValueNegate(const Value *v1);
    Value *ValueSubtract(const Value *v1, const Value *v2);
    Value *ValueReverseSubtract(const Value *v1, const Value *v2);
    Value *ValueReverseMultiply(const Value *v1, const Value *v2);
    Value *ValueDivide(const Value *v1, const Value *v2);
    Value *ValueReverseMultiply(const Value *v1, const Value *v2);
    void ValuePrint(const Value *v1);
    void ValueFree(Value *v1);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _C_ENGINE_HEADER_ */