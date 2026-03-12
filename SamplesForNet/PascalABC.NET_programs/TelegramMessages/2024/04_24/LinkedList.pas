type 
  Node = auto class
    data: integer;
    next: Node;
  end;
  
procedure For_each(n: Node; act: procedure(x: integer));
begin
  while n <> nil do
  begin
    act(n.data);
    n := n.next;
  end;
end;  
  
begin
  var n: Node := nil;
  n := new Node(5,n);
  n := new Node(2,n);
  n := new Node(3,n);
  Println(n);
  For_each(n,x -> Print(x));
end.  