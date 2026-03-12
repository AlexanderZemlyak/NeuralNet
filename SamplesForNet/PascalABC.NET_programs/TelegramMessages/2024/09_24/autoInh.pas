type
  Base = class
    procedure Output := Print(Self);
  end;
  Point = auto class(Base)
    x,y: integer;
  end;
  
begin
  var p := new Point(2,3);
  p.Output;
end.  