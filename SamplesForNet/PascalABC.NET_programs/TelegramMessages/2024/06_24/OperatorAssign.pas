type A = class
public  
  y: integer;
  static procedure operator:=(Self: A; y: integer);
  begin
    Self.y := y;
  end;
end;

begin
  var a1 := new A;
  a1 := 555;
  Print(a1);
end.