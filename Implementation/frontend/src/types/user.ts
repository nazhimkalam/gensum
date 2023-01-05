import { DomainType } from "../enum/DomainType";

export type User = {
  id: string | undefined;
  displayName: string | undefined;
  email: string | undefined;
  domainName: string | undefined;
  domainType: DomainType | undefined;
  isAccessible: boolean | undefined;
};

