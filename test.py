from DecisionTree import DecisionTree

file1 = "learn_test.csv"
file2 = "test_example.csv"

tree = DecisionTree()
tree.create_tree(file1, (4, 'Walk'))
tree.generate_rules()
# tree.use_tree(file2)
print(tree.rules)
