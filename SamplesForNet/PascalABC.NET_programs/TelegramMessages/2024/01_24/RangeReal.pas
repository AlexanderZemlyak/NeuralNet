begin
  Range(1,3,0.5).Println;
  Range(1,2,1/3).Println; // правая граница всегда входит несмотря на погрешность округления
  Range(3,1,-0.2).Println;
  Range(1,2,-0.2).Println; // пустая
  Range(2,1,0.2).Println;  // пустая
  Range(1,3,0.3).Println;  // не до конца
end.