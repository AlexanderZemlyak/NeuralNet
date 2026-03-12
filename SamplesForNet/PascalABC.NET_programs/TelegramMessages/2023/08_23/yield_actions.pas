function ff: sequence of integer;
begin
  Print(111);
  yield 1;
  Print(222);
  yield 2;
end;

function gg: sequence of integer;
begin
  Print(333);
  yield 3;
  Print(444);
  yield 4;
end;

begin
  var q: IEnumerator<integer> := ff.GetEnumerator;
  var p: IEnumerator<integer> := gg.GetEnumerator;
  Print(q.Current);
  while q.MoveNext do
  begin  
    p.MoveNext
  end;  
end.