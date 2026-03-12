type 
  IPrintable = interface
    procedure Print;
  end;
  Point = class(IPrintable)
    x,y: real;
  public  
    constructor (xx,yy: real) := (x,y) := (xx,yy);
    procedure Print := PABCSystem.Println($'Координаты: ({x},{y})');
  end;
  Person = class(IPrintable)
    name: string;
  public  
    constructor(n: string) := name := n;
    procedure Print := PABCSystem.Println($'Имя: {name}');
  end;
  
begin
  var L := new List<IPrintable>;
  L.Add(new Point(2,3));
  L.Add(new Person('Иван'));
  foreach var obj in L do
    obj.Print;
end.
  
  