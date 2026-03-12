label 1;

begin
  var Спичек := 15;
  Println($'Спичек на столе:',Спичек);
  repeat
    1:
    var Ход := ReadInteger('Возьмите от 1 до 3 спичек, но не более чем на столе:');
    if Ход not in 1..3 then goto 1;
    if Ход > Спичек then goto 1;
    
    Спичек -= Ход;
    Println($'Спичек на столе:',Спичек);
  until Спичек = 0;
end.