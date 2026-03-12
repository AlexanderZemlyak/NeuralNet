type
  Node<T> = auto class
    data: T;
    left,right: Node<T>;
  end;
  
procedure AddToTree(var root: Node<integer>; x: integer);
begin
  if root = nil then
    root := new Node<integer>(x,nil,nil)
  else if x < root.data then
    AddToTree(root.left,x)
  else AddToTree(root.right,x)
end;

procedure PrintTree(root: Node<integer>);
begin
  if root = nil then exit;
  PrintTree(root.left);
  Print(root.data);
  PrintTree(root.right)
end;
  
begin
  var root: Node<integer> := nil;
  loop 30 do
    AddToTree(root,Random(100));
  PrintTree(root);
end.