type 
  MyArr<T> = class
  private
    a: array of T;
  public
    constructor(aa: array of T) := a := aa;
  end;
  NumPabc = static class
  public
    static function &Array<T>(aa: array of T): MyArr<T> := new MyArr<T>(aa);
  end;

function Si(x: real) := Sin(x);

function Sin(a: array of real): array of real;
begin
  Result := a.Select(Si).ToArray;  
end;

function operator>(a,b: array of real): array of boolean; extensionmethod;
begin
  Result := a.Zip(b,(x,y)->x>y).ToArray;  
end;


begin
  var x := |1.0,2.0,3.0|;
  var y := |5.0,3.0,1.0,5.0|;
  Print(x > y);
end.