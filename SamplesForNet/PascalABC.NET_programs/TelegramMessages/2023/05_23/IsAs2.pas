type
  Shape = class
  public  
    x,y: real;
    constructor(xx,yy: real) := (x,y) := (xx,yy);
  end;
  Point = class(Shape)
  end;
  Circle = class(Shape)
  public  
    r: real;
    constructor(x,y,rr: real);
    begin
      inherited Create(x,y);
      r := rr;
    end;
    procedure SetRadius(rr: real) := r := rr;
  end;
  
begin
  var L := new List<Shape>;
  L.Add(new Point(7,12));
  L.Add(new Circle(20,20,5));
  L.Add(new Point(10,30));
  Println(L);
  foreach var obj in L do
    if obj is Circle (var c) then
      c.SetRadius(777);
  Println(L);
end.  