#include "../includes/engine.h";
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

Value *ValueNew(double data)
{
    Value *v = (Value *)malloc(sizeof(Value));
    v->data = data;
    v->grad = 0;
    v->backward = NULL;
    v->num_prev = 0;
    return v;
}

Value *ValueAdd(const Value *v1, const Value *v2)
{
    Value *result = ValueNew(v1->data + v2->data);
    result->_prev[0] = (Value *)v1;
    result->_prev[1] = (Value *)v2;
    result->num_prev = 2;
    result->backward = &ValueAddBackward;
    return result;
}

void ValueAddBackward(Value *v1)
{
    v1->_prev[0]->grad += v1->grad;
    v1->_prev[1]->grad += v1->grad;
}

Value *ValueMultiply(const Value *v1, const Value *v2)
{
    Value *result = ValueNew(v1->data * v2->data);
    result->_prev[0] = (Value *)v1;
    result->_prev[1] = (Value *)v2;
    result->num_prev = 2;
    result->backward = &ValueMultiplyBackward;
    return result;
}

void ValueMultiplyBackward(Value *v1)
{
    v1->_prev[0]->grad += v1->grad * v1->_prev[1]->data;
    v1->_prev[1]->grad += v1->grad * v1->_prev[0]->data;
}

Value *ValuePower(const Value *v1, const Value *exponent)
{
    Value *result = ValueNew(pow(v1->data, exponent->data));
    result->_prev[0] = (Value *)v1;
    result->_prev[1] = (Value *)exponent;
    result->num_prev = 2;
    result->backward = &ValuePowerBackward;
    return result;
}

void ValuePowerBackward(Value *v1)
{
    v1->_prev[0]->grad += v1->_prev[1]->data * pow(v1->_prev[0]->data, v1->_prev[1]->data - 1) * v1->grad;
}

Value *ValueReLU(const Value *v1)
{
    double relu_value = (v1->data > 0) ? v1->data : 0;
    Value *result = ValueNew(relu_value);
    result->_prev[0] = (Value *)v1;
    result->num_prev = 1;
    result->backward = &ValueReLUBackward;
    return result;
}

void ValueReLUBackward(Value *v1)
{
    v1->_prev[0]->grad += (v1->_prev[0]->data > 0) * v1->grad;
}

Value *ValueTanh(const Value *v1)
{
    double x = pow(EULER_CONST, 2 * v1->data);
    Value *result = ValueNew(x - 1 / x + 1);
    result->_prev[0] = (Value *)v1;
    result->num_prev = 1;
    result->backward = &ValueTanhBackward;
    return result;
}

void ValueTanhBackward(Value *v1)
{
    v1->_prev[0]->grad += (1 - pow(v1->data, 2)) * v1->grad;
}

void TopologicalSort(Value *self, bool *visited, Value **topo, int *index)
{
    visited[self->index] = true;

    for (int i = 0; i < self->_prev_size; ++i)
    {
        if (!visited[self->_prev[i].index])
        {
            TopologicalSort(&(self->_prev[i]), visited, topo, index);
        }
    }

    topo[(*index)++] = self;
}

void ValueBackward(Value *self)
{
    bool visited[self->_prev_size];
    for (int i = 0; i < self->_prev_size; ++i)
    {
        visited[i] = false;
    }
    Value *topo[self->_prev_size];
    int index = 0;
    TopologicalSort(self, visited, topo, &index);
    if (self->backward != NULL)
    {
        self->backward(self);
    }
    for (int i = self->_prev_size - 1; i >= 0; --i)
    {
        ValueBackward(&(topo[i]));
    }
}

Value *ValueNegate(const Value *v1)
{
    Value *result = ValueNew(-(v1->data));
    result->_prev[0] = (Value *)v1;
    result->num_prev = 1;
    return result;
}

Value *ValueSubtract(const Value *v1, const Value *v2)
{
    Value *neg_v2 = ValueNegate(v2);
    Value *result = ValueAdd(v1, neg_v2);
    free(neg_v2);
    return result;
}

Value *ValueReverseSubtract(const Value *v1, const Value *v2)
{
    return ValueSubtract(v2, v1);
}

Value *ValueReverseMultiply(const Value *v1, const Value *v2)
{
    return ValueMultiply(v2, v1);
}

Value *ValueDivide(const Value *v1, const Value *v2)
{
    Value *result = ValueMultiply(v1, ValuePower(v2, -1.0));
    return result;
}

Value *ValueReverseDivide(const Value *v1, const Value *v2)
{
    return ValueDivide(v2, v1);
}

void ValuePrint(const Value *v1)
{
    printf("Value(data=%.2f, grad=%.2f)\n", v1->data, v1->grad);
}