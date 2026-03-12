type Node<T> = auto class
  Data: T;
  Next: Node<T>;
end;

procedure AddFirst<T>(var first: Node<T>; x: T) := first := new Node<T>(x,first);

begin
  var lst: Node<integer> := nil;
  AddFirst(lst,3);
  AddFirst(lst,5);
  AddFirst(lst,4);
  Println(lst);
  
end.

