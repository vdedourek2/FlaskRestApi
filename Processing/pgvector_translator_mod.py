from typing import Dict, Tuple, Union

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)

# pgvector comparators and operators
# IN, NIN, BETWEEN, GT, LT, NE = "in", "nin", "between", "gt", "lt", "ne"
#  EQ, LIKE, CONTAINS, OR, AND = "eq", "like", "contains", "or", "and"
# https://github.com/langchain-ai/langchain/issues/9726
# https://github.com/langchain-ai/langchain/issues/13281

def flatten_filter_structure(filter_structure)->list:
    """
    Translate dictionary structure to simple list structure.
    {....} -> [attribute:{comparator:value}, ...]
    Result is sorted by key.
    """
    result = []

    def flatten_helper(structure):
        for key, value in structure.items():
            if key == 'and':
                for sub_filter in value:
                    flatten_helper(sub_filter)
            else:
                result.append({key: value})

    flatten_helper(filter_structure)
    
    sorted_result = sorted(result, key=lambda x: list(x.keys())[0])

    return sorted_result

def process_filter_operations(filter_list)->list:
    """
    Replacing.  lte -> lt, gte -> gt with value modification
    """    
    result = []

    for item in filter_list:
        key, item2 = list(item.items())[0]
        operation, value = list(item2.items())[0]

        if operation in ('gte', 'lte') and (isinstance(value, int) or isinstance(value, float)):
            difference = 1 if isinstance(value, int) else 0.1   # 1 for int, 0.1 for float
            value = value - difference if operation == 'gte' else value + difference
            operation = operation[:2]                           # only gt ot lt
 
        result.append({key: {operation: value}})

    return result

def combine_adjacent_operations(filtered_list)->list:
    """
    Pairs (key:{'lt': value1}, key:{'gt': value2}) for the same key are replaced by {'between', [value1, value2]}
    """   
    combined_result = []
    last_item = {"":{"":0}}

    for item in filtered_list:
        key, item2 = list(item.items())[0]
        operation, value = list(item2.items())[0]        
        last_key = list(last_item.keys())[0]
            
        if key == last_key:
            last_value_item = list(last_item.values())[0]
            last_operation, last_value = list(last_value_item.items())[0] 

            if last_operation == 'gt' and operation =='lt':                             # item1 > val1 and item2 < val2
                combined_result.pop()   # removing last item
                combined_result.append({key: {'between': [last_value, value]}})   # between val1 and val2
            elif last_operation == 'lt' and operation =='gt':                           # item1 < val1 and item2 > val2
                combined_result.pop()   # removing last item
                combined_result.append({key: {'between': [value, last_value]}})   # between val2 and val1
            else:
                combined_result.append(item)
        else:
            last_item = item
            combined_result.append(item)

    return combined_result


class PgvectorTranslator(Visitor):
    """Translate `pgvector` internal query language elements to valid filters."""

    allowed_operators = [Operator.AND]
 
    """Subset of allowed logical operators."""
    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.LIKE,
        Comparator.IN,
        Comparator.NIN,
    ]
    """Subset of allowed logical comparators."""

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        return f"{func.value}"

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        return {self._format_func(operation.operator): args}

    def visit_comparison(self, comparison: Comparison) -> Dict:
       return {
            comparison.attribute: {
                self._format_func(comparison.comparator): comparison.value
            }
        }

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            filter_structure = structured_query.filter.accept(self)
            
            #  post processing
            filtered_list = flatten_filter_structure(filter_structure)
            processed_list = process_filter_operations(filtered_list)
            combined_list = combine_adjacent_operations(processed_list)            

            kwargs = {"filter": combined_list}
        
            return structured_query.query, kwargs
