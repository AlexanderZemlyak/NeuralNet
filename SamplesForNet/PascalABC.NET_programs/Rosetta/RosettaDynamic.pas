uses System.Dynamic;

type 
  Example = class(DynamicObject)
    public 
    function TryInvokeMember(binder: InvokeMemberBinder; args: array of Object;
      var res: object): boolean; override;
    begin
      res := nil;
      Print(binder.Name);
      Result := True;
    end;
  end;

begin
  var ex := new Example;
  
end.