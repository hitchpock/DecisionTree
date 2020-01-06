import csv
from math import log2


class DecisionTree():
    """
    Дерево принятия решений. Работает на основе алгоритма ID3.

    :param self.dt: словарь содержащий правила
    :param self.rules: строка с читабельным видом правил
    :param self.new_data: список содержащий новые данные, результат применения правил на сырые данные

    :return: [description]
    :rtype: [type]
    """
    def __init__(self):
        self.new_data = []

    def create_tree(self, filename, target_attribute):
        """
        Метод обработки обучающего набора данных и создание словаря правил.

        :param filename: имя файла с обучающим набором
        :type filename: str
        :param target_attribute: целевой атрибут таблицы
        :type target_attribute: tuple(number_attribute, name_attribute)
        """
        title, rows = parse_csv(filename)
        title = [(n, v) for n, v in enumerate(title)]
        rows = add_types(rows)
        attribute_values = attr_dict(rows, title)

        attributes = [x for x in title if x != target_attribute]

        self.dt = id3(rows, target_attribute, attributes, attribute_values)

    def generate_rules(self):
        """
        Создание строки вывода правил

        :return: строка правил
        :rtype: str
        """
        def dfs(tree, depth):
            tab = " " * (depth * 4)
            rules = []
            if len(tree.keys()) != 1:
                attribute_name = tree['test_attributes']
                for k, v in tree.items():
                    if k != 'test_attributes':
                        rules.append(tab + "if '{0}' == '{1}':".format(attribute_name, k))
                        block = dfs(v, depth + 1)
                        rules += block if isinstance(block, list) else [block]
            else:
                # Почему то не все значения целевых атрибутов являются tuple
                if isinstance(tree[list(tree.keys())[0]], tuple):
                    rules.append(tab + "return '{0}'".format(tree[list(tree.keys())[0]][1]))
                else:
                    rules.append(tab + "return '{0}'".format(tree[list(tree.keys())[0]]))
            return rules
        self.rules = "\n".join(dfs(self.dt, 0))
        return self.rules

    def use_tree(self, filename):
        """
        Применение правил к сырым данным.

        :param filename: имя файла с данными
        :type filename: str
        """
        title, rows = parse_csv(filename)
        data = hashmap(rows, title)
        target_attribute = result_attribute(self.dt)
        for row in data:
            self.new_data.append(parse_dict(self.dt, row, target_attribute))


def add_types(rows):
    """
    Присвоение порядкового номера всем значениям атрибутов.

    :param rows: список примеров обучающего набора
    :type rows: list(list)
    :return: список обучающего набора
    :rtype: list(list)
    """
    return [[(i, v) for i, v in enumerate(row)] for row in rows]


def attr_dict(rows, title):
    """
    Создание множества значений для каждого атрибута.

    :param rows: список примеров обучающего набора
    :type rows: list(list)
    :param title: список атрибутов
    :type title: list
    :return: словарь, ключ - атрибут, значения - множество значений атрибута
    :rtype: dict
    """
    attribute_values = {}
    for n, t in enumerate(title):
        attribute_values[t[1]] = set()
        for row in rows:
            attribute_values[t[1]].add(row[n][1])
    return attribute_values


def parse_csv(filename):
    """
    Парсинг csv файла.

    :param filename: имя csv файла
    :type filename: string
    :return: title - список аргументов, rows - список строк
    :rtype: title - list, rows - list(list)
    """
    rows = []
    with open(filename) as f:
        reader = csv.reader(f, skipinitialspace=True)
        # title - хранит список заголовков
        title = next(reader)
        for r in reader:
            # r - строка: список характеристик
            # rows - строки: двумерный массив, список списков строк
            rows.append(r)
    return title, rows


def check_all_same(examples, target_attribute):
    """
    Проверка. Если все примеры по заданному атрибуту имеют одинаковое значение, то возвращаем True.

    :param examples: Двумерный массив с обучающим набором
    :type examples: двумерный массив
    :param target_attribute: целевой атрибут
    :type target_attribute: tuple
    :param title: список атрибутов
    :type title: str
    :return: True/False
    :rtype: Bool
    """
    mark = examples[0][target_attribute[0]][1]
    for row in examples:
        if row[target_attribute[0]][1] != mark:
            return False
    return True


def find_most_common(examples, target_attribute):
    """
    Возвращаем наиболее часто встречающееся значение атрибута.

    :param examples: список примеров из обучающего набора
    :type examples: двумерный массив
    :param target_attribute: целевой атрибут
    :type target_attribute: tuple
    :return: часто встречающееся значение
    :rtype: string
    """
    dct = {}
    for example in examples:
        dct[example[target_attribute[0]]] = dct.get(example[target_attribute[0]], 0) + 1
    res = sorted(list(dct.items()), key=lambda x: x[1], reverse=True)
    #print(res)
    return res[0][0][1]


def filter_examples(examples, attribute, value):
    """
    Фильтруем список примеров, и оставляем те, у которых значение атрибута "attribute" равно "value".

    :param examples: список примеров из обучающего набора
    :type examples: двумерный массив
    :param attribute: атрибут таблицы
    :type attribute: tuple
    :param value: значение атрибута с котрым сравниваем
    :type value: string
    :return: отфильтрованный список
    :rtype: list
    """
    filtered = []
    for example in examples:
        if example[attribute[0]][1] == value:
            filtered.append(example)
    return filtered


def calc_entropy(examples, target_attribute):
    """
    Вычисление энтропии для списка примеров по заданному атрибуту.

    :param examples: список примеров из обучающего набора
    :type examples: двумерный массив
    :param target_attribute: целевой атрибут
    :type target_attribute: tuple
    :return: энтропия
    :rtype: float
    """
    dct = {}
    ln = len(examples)
    for example in examples:
        dct[example[target_attribute[0]]] = dct.get(example[target_attribute[0]], 0) + 1
    x = 0
    for k in dct.keys():
        x += float(dct[k]/ln) * log2(float(dct[k]/ln))
    return -x


def calc_info_gain(examples, test_attribute, target_attribute):
    """
    Вычисляем информационный прирост списка тренировочных случаев в соответсвии с проверочным атрибутом и целевым атрибутом.

    :param examples: список примеров из обучающего набора
    :type examples: двумерный массив
    :param test_attribute: проверочный атрибут
    :type test_attribute: tuple
    :param target_attribute: целевой атрибут
    :type target_attribute: tuple
    :return: информационный прирост
    :rtype: int
    """
    groups = {}
    values = []
    for example in examples:
        v = example[test_attribute[0]]
        group = groups.get(v)
        if group is None:
            groups[v] = [example]
            values.append(v)
        else:
            groups[v].append(example)
    g = calc_entropy(examples, target_attribute)
    for v in values:
        group = groups[v]
        p = float(len(group)/len(examples))
        h = calc_entropy(group, target_attribute)
        g -= p*h
    return g


def best_classifier(examples, attributes, target_attribute):
    """
    Наилучший классификационный атрибут.

    :param examples: список примеров из обучающего набора
    :type examples: двумерный массив
    :param attributes: список атрибутов
    :type attributes: list
    :param target_attribute: целевой атрибут
    :type target_attribute: tuple
    :return: наилучший атрибут
    :rtype: string
    """
    maximum = 0
    best_attr = None
    for attribute in attributes:
        gain = calc_info_gain(examples, attribute, target_attribute)
        if gain >= maximum:
            maximum = gain
            best_attr = attribute
    return best_attr


def id3(examples, target_attribute, attributes, attribute_values):
    """
    Алгоритм id3.

    :param examples: список примеров из обучающего набора
    :type examples: двумерный массив
    :param target_attribute: целевой атрибут
    :type target_attribute: tuple
    :param attributes: список оставшихся атрибутов
    :type attributes: list
    :param attribute_values: словарь, содержащий допустимые значения атрибутов
    :type attribute_values: dict
    :return: словарь(дерево)
    :rtype: dict
    """
    dt = {}
    if check_all_same(examples, target_attribute):
        ex = examples[0][target_attribute[0]]
        dt[target_attribute[1]] = ex
        return dt
    if len(attributes) < 1:
        mc = find_most_common(examples, target_attribute)
        dt[target_attribute[1]] = mc
        return dt
    a = best_classifier(examples, attributes, target_attribute)
    dt['test_attributes'] = a[1]
    for v in attribute_values[a[1]]:
        examples_subset = filter_examples(examples, a, v)
        if len(examples_subset) < 1:
            mc = find_most_common(examples, target_attribute)
            c = {}
            c[target_attribute[1]] = mc
            dt[v] = c
        else:
            offspring_attributes = [x for x in attributes if x != a]
            c = id3(examples_subset, target_attribute, offspring_attributes, attribute_values)
            dt[v] = c
    return dt


def hashmap(rows, title):
    """
    Преобразование простых строк в словари.

    :param rows: список строк с данными
    :type rows: list(list)
    :param title: список атрибутов
    :type title: list
    :return: список словарей
    :rtype: list
    """
    res_lst = []
    for row in rows:
        res_lst.append(dict(zip(title, row)))
    return res_lst


def result_attribute(dct):
    """
    Достаем целевой атрибут из словаря правил.

    :param dct: словарь правил
    :type dct: dict
    :return: целевой атрибут
    :rtype: str
    """
    keys_list = [k for k in dct.keys() if k != 'test_attributes']
    for k in keys_list:
        if isinstance(dct[k], tuple):
            return k
        else:
            return result_attribute(dct[k])


def parse_dict(rules_dict, data, target_attribute):
    """
    Вычисление значения целевого атрибута.

    :param rules_dict: словарь правил
    :type rules_dict: dict
    :param data: словарь характеристик
    :type data: dict
    :param target_attribute: целевой атрибут
    :type target_attribute: str
    :return: результат
    :rtype: lst
    """
    if target_attribute not in rules_dict.keys():
        test_attribute = rules_dict['test_attributes']
        char = data[test_attribute]
        return parse_dict(rules_dict[char], data, target_attribute)
    else:
        return rules_dict[target_attribute]
