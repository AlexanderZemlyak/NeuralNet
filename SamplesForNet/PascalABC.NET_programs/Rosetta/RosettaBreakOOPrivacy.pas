// https://rosettacode.org/wiki/Break_OO_privacy#PascalABC.NET

uses System.Reflection;

type MyClass = class
private
  x: integer := 5;
public 
  procedure PrintFields := Print(x);
end;

begin
  var a := new MyClass;
  var fi := a.GetType.GetField('x', BindingFlags.Instance or BindingFlags.NonPublic);
  fi.SetValue(a,777);
  Print(a);
end.