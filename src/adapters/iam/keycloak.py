from typing import Dict, Any
import requests
from .base import AccessControlBase

class KeycloakAccessControl(AccessControlBase):
    def get_admin_access_token(self, keycloak_id, keycloak_passwrod, endpoint):
        url = f"{endpoint}/realms/master/protocol/openid-connect/token"
        data = {
            "client_id": "admin-cli",
            "username": keycloak_id,
            "password": keycloak_passwrod,  # key name kept as-is
            "grant_type": "password"
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        resp = requests.post(url, data=data, headers=headers)
        resp.raise_for_status()
        return resp.json().get("access_token")

    def find_user_id_by_username(self, username, admin_token, realm, endpoint):
        url = f"{endpoint}/admin/realms/{realm}/users"
        headers = {"Authorization": f"Bearer {admin_token}"}
        params = {"username": username}

        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        users_found = resp.json()

        if not users_found:
            return None

        return users_found[0]["id"]

    def find_resource_id_by_name(self, resource_name, admin_token, realm, endpoint, client_uuid):
        url = f"{endpoint}/admin/realms/{realm}/clients/{client_uuid}/authz/resource-server/resource"
        headers = {"Authorization": f"Bearer {admin_token}"}

        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        for resource in resp.json():
            if resource.get("name") == resource_name:
                return resource.get("_id") or resource.get("id")

        return None

    def policy_evaluate(self, user_id, resource_id, resource_name, admin_token, realm, endpoint, client_uuid):
        evaluate_url = f"{endpoint}/admin/realms/{realm}/clients/{client_uuid}/authz/resource-server/policy/evaluate"
        headers = {"Authorization": f"Bearer {admin_token}", "Content-Type": "application/json"}
        data = {
            "userId": user_id,
            "clientId": client_uuid,
            "entitlements": False,
            "resources": [{"_id": resource_id, "name": resource_name, "scopes": [{"name": "read"}]}]
        }

        resp = requests.post(evaluate_url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json().get("status") == "PERMIT"

    def check_access(self, resource_metadata: Dict[str, Any]) -> bool:
        try:
            keycloak_id = self.user_credential.get("keycloak_admin_id")
            keycloak_passwrod = self.user_credential.get("keycloak_admin_passwrod")  # kept as-is
            keycloak_user_name = self.user_credential.get("keycloak_user_name")
            
            if not (keycloak_id and keycloak_passwrod and keycloak_user_name):
                return False

            realm = (resource_metadata.get("parameters") or {}).get("realm")
            client_uuid = (resource_metadata.get("parameters") or {}).get("client_uuid")
            object_name = resource_metadata.get("file_name")
            endpoint = resource_metadata.get("endpoint")

            admin_token = self.get_admin_access_token(keycloak_id, keycloak_passwrod, endpoint)
            if not admin_token:
                return False

            user_id = self.find_user_id_by_username(keycloak_user_name, admin_token, realm, endpoint)
            if not user_id:
                return False

            resource_id = self.find_resource_id_by_name(object_name, admin_token, realm, endpoint, client_uuid)
            if not resource_id:
                return False

            return self.policy_evaluate(user_id, resource_id, object_name, admin_token, realm, endpoint, client_uuid)
        except Exception:
            return False
