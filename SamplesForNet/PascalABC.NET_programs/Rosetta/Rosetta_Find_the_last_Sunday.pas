// https://rosettacode.org/wiki/Find_the_last_Sunday_of_each_month#PascalABC.NET

function LastSunday(month,year: integer): DateTime;
begin
  var date := new DateTime(year,month,DateTime.DaysInMonth(year, month));
  while date.DayOfWeek <> System.DayOfWeek.Sunday do
    date := date.AddDays(-1);
  Result := Date;
end;

begin
  for var month:=1 to 12 do
    Println(LastSunday(month,2024).ToString('yyyy-MM-dd'));
end.
