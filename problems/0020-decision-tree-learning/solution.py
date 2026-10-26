import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    # Your code here
    #formula for entropy is sum - p log2 p
    freq={}
    n=len(labels)
    if n==0:
        return float("inf")
    for elm in labels:
        freq[elm]=freq.get(elm,0)+1
    entropy=0
    for u,v in freq.items():
        prob=v/n 
        entropy+= (-1)* prob*math.log2(prob)
    return entropy
    pass

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    # Your code here
    target_values=[ex[target_attr] for ex in examples]
    parent_entropy=calculate_entropy(target_values)
    values=sorted(set( ex[attr] for ex in examples))
    n=len(examples)
    child_entropy=0
    for val in values:
        subset=[
            ex for ex in examples if ex[attr]==val
        ]
        n_subset=len(subset)
        labels=[ ex[target_attr] for ex in subset]
        entropy=calculate_entropy(labels)
        child_entropy+=entropy*(n_subset/n)
    return parent_entropy-child_entropy
    pass

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    # Your code here
    freq={}
    for ex in examples:
        val=ex[target_attr]
        freq[val]=freq.get(val,0)+1
    
    max_freq=max(freq.values())

    values=[key for key,val in freq.items() if val==max_freq]
    return min(values)
    pass

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    # Your code here
    # if all labels are same return that -- stopping condition for recursive call 
    labels=[ex[target_attr] for ex in examples]
    if len(set(labels))==1:
        return labels[0]
    # if attributes len is 0  return majority_class
    if len(attributes)==0 :
        return majority_class(examples,target_attr)
    best_ig=-1
    best_feature=None
    for att in attributes:
        ig=calculate_information_gain(examples,att,target_attr)
        if ig>best_ig:
            best_ig=ig
            best_feature=att
    
    #if no attribute give such division
    if best_ig<0:
        return majority_class(examples,target_attr)
    
    rem_att=[att for att in attributes if att!=best_feature]

    tree={best_feature:{}}

    #extract unique value for that attributes 
    values=sorted(set(ex[best_feature] for ex in examples))
    
    for val in values:
        subset=[
            ex for ex in examples if ex[best_feature]==val

        ]
        tree[best_feature][val]=learn_decision_tree(subset,rem_att,target_attr)
    
    return tree

    pass