type
  Shape = class
  public  
    x,y: real;
    constructor(xx,yy: real) := (x,y) := (xx,yy);
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
  var sh: Shape := new Circle(200,200,100);
  Println(sh);
  // sh.SetRadius(500); // Неизвестное имя 'SetRadius'
  (sh as Circle).SetRadius(500);
  Println(sh);
end.  