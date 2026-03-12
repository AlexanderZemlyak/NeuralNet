begin
  var f := OpenRead('finally.pas');
  try
    try
        f.ReadString.Println;
        f.ReadString.ToInteger.Println;
    finally
      f.Close;
      Println('Файл закрыт!')
    end;  
  except
    on e: Exception do
      Println(e.Message,e.GetType);
  end;  
end.