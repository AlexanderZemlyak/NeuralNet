// https://rosettacode.org/wiki/Date_format#

begin
  var today := DateTime.Now.Date;
  Println(today.ToString('yyyy-MM-dd'));
  Println($'{today:D}');
end.