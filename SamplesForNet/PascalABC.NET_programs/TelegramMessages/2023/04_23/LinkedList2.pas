type Node<T> = auto class
  Data: T;
  Next: Node<T>;
end;

procedure AddFirst<T>(var first: Node<T>; x: T) := first := new Node<T>(x,first);

procedure &Foreach<T>(Self: Node<T>; act: T->()); extensionmethod := 
  while Self<>nil do
  begin
    act(Self.Data);
    Self := Self.Next;
  end;

begin
  var lst: Node<integer> := nil;
  AddFirst(lst,3);
  AddFirst(lst,5);
  AddFirst(lst,4);
  Println(lst);
  lst.Foreach(x -> Print(x));
end.

