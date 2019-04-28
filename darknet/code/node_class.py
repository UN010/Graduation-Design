# encoding=utf-8


class Node(object):
    """节点类"""
    def __init__(self, item):
        self.item = item
        self.next = None
    def __repr__(self):
        return str(self.item)
class CycleSingleLinkList(object):
    """循环单链表"""
    def __init__(self):
        # 指向头节点
        node1 = Node("green")
        node2 = Node("yellow")
        node3 = Node("red")
        node4 = Node("red")
        self._head = node1
        node1.next = node2
        node2.next = node3
        node3.next = node4
        node4.next = node1


if __name__ == '__main__':
    ll = CycleSingleLinkList()
    z=ll._head
    z=z.next
    print(z)
    # for i in range(0,2):
    # 	print(ll.__head)
    # 	z=ll.__head
    # 	print(z.next)
