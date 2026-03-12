procedure PrintPecents(n,count: integer);
begin
  loop count do
  begin
    var (Орлы, Решки) := (0,0);
    loop n do
      if Random(0,1)=0 then
        Орлы += 1;
    Решки := n - Орлы;
    Print(Орлы/n*100);
  end;
  Println
end;

begin
  PrintPecents(100,10);
  PrintPecents(1000,10);
  PrintPecents(10000,10);
  PrintPecents(100000,10);
end.