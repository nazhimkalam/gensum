import React, { useEffect, useState } from "react";
import styled from "styled-components";
import { Button, Form, Input, Select, Typography } from "antd";
import { DomainType } from "../../enum/DomainType";
import { useSelector } from "react-redux";
import { selectUser } from "../../redux/reducers/userReducer";
import { db } from "../../firebase/firebase";

const UserProfile = () => {
  const { Option } = Select;
  const { Title } = Typography;
  const user = useSelector(selectUser);

  const [username, setUsername] = useState("");
  const [domainType, setDomainType] = useState("");
  const [accessTypeAttribute, setAccessTypeAttribute] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const userId = user?.id;
    setIsLoading(true);

    db.collection("users")
      .doc(userId)
      .get()
      .then((doc: any) => {
        if (doc.exists) {
          const data = doc.data();
          setUsername(data.name);
          setDomainType(data.type);
          setAccessTypeAttribute(data.isAccessible);
        } else {
          console.log("No such document!");
        }
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [user]);

  const handleFormSubmit = (e: any) => {
    e.preventDefault();

    if (parseInt(domainType) === 0 || username === "") {
      alert("Please fill in all the fields");
      return;
    }

    const userId = user?.id;
    setIsLoading(true);

    db.collection("users")
      .doc(userId)
      .update({
        name: username,
        type: domainType,
        isAccessible: accessTypeAttribute,
      })
      .then(() => {
        alert("User profile updated successfully");
      })
      .finally(() => {
        setIsLoading(false);
      });
  };

  return (
    <StyledContainer>
      <Form>
        <Title level={2}>User Profile</Title>
        <br />
        <Form.Item className="form-item" label="Username">
          <Input
            placeholder="Enter username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={isLoading}
          />
        </Form.Item>
        <Form.Item className="form-item" label="Domain Type">
          <Select
            placeholder="Select domain type"
            value={domainType}
            onChange={(value) => setDomainType(value)}
            disabled={isLoading}
          >
            <Option value={0}> Select domain type </Option>
            <Option value={DomainType.MOVIE_THEATER}> Movie Theater </Option>
            <Option value={DomainType.ECOMMERCE}> Ecommerce </Option>
            <Option value={DomainType.HOTEL}> Hotel </Option>
            <Option value={DomainType.RESTURANT}> Resturant </Option>
          </Select>
        </Form.Item>
        <Form.Item className="form-item" label="Access Type Attribute">
          <Select
            placeholder="Select access type attribute"
            value={accessTypeAttribute}
            disabled={isLoading}
            onChange={(value) => setAccessTypeAttribute(value)}
          >
            <Option value={true}>Yes</Option>
            <Option value={false}>No</Option>
          </Select>
        </Form.Item>

        <Form.Item className="form-item">
          <Button className="btn btn-primary" onClick={handleFormSubmit} 
            disabled={isLoading}
          >
            {!isLoading ? 'Update profile' : 'Loading...'}
          </Button>
        </Form.Item>
      </Form>
    </StyledContainer>
  );
};

export default UserProfile;

const StyledContainer = styled.div`
  margin: 1pc 10vw;

  > form {
    margin: 5pc 0;

    > h1 {
      margin-bottom: 1pc;
    }

    .form-item {
      margin-bottom: 2pc;
    }

    button {
      border: 1px black solid;
      color: black;
    }
  }
`;
