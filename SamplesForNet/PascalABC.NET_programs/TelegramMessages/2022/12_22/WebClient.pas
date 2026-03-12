uses System.Net;

begin
  var wc := new WebClient;
  wc.Encoding := Encoding.UTF8;
  var s := wc.DownloadString('http://pascalabc.net/ssyilki-dlya-skachivaniya');
  var mm := s.Matches('версия (\d*\.\d*\.\d*), сборка (\d*) от (\d*\.\d*\.\d*)');
  if mm.Count > 0 then
  begin
    var f := mm.First;
    Println('Версия:',f.Groups[1].Value);
    Println('Сборка:',f.Groups[2].Value);
    Println('Дата сборки:',f.Groups[3].Value)
  end;
end.