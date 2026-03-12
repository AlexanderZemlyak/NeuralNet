// https://rosettacode.org/wiki/Doubly-linked_list/Traversal#PascalABC.NET

begin
  var lst := new LinkedList<integer>(1..10);
  var current := lst.First;
  while current <> nil do
  begin
    Print(current.Value);
    current := current.Next;
  end;
end.
