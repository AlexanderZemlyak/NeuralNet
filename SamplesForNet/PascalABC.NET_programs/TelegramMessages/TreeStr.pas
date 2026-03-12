type 
  Node = auto class
    data: string;
    left,right: Node;
  end;

function CreateTree(n: integer; s: string): Node := 
  if n = 0 then 
    nil
  else new Node(s,CreateTree((n-1) div 2,s+'0'),CreateTree(n-1-(n-1) div 2,s+'1'));

begin
  Print(CreateTree(16,''));  
end.