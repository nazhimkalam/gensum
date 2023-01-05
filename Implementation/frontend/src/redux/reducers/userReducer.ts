import { User } from "./../../types/user";
import { createSlice } from "@reduxjs/toolkit";

const userState: User = {
  id: undefined,
  displayName: undefined,
  email: undefined,
  domainName: undefined,
  domainType: undefined,
  isAccessible: undefined,
};

export const userSlice = createSlice({
  name: "user",
  initialState: {
    user: userState,
  },
  reducers: {
    login: (state, action) => {
      state.user = action.payload;
    },
    logout: (state) => {
      state.user = userState;
    },
  },
});

export const { login, logout } = userSlice.actions;
export const selectUser = (state: any) => state.user.user;
export default userSlice.reducer;
