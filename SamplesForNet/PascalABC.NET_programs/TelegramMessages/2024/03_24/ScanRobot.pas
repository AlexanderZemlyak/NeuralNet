type Point = auto class x,y:integer end;

function NewPos(pos: Point; dir: char): Point;
begin
  case dir of
    '>': Result := new Point(pos.x+1,pos.y);
    '<': Result := new Point(pos.x-1,pos.y);
    '^': Result := new Point(pos.x,pos.y-1);
    'v': Result := new Point(pos.x,pos.y+1);
  end;  
end;

begin
  var initialpos := new Point(2,3);
  '>>^>>v<'.Scan(initialpos,(pos,dir)->NewPos(pos,dir)).Println;
end.