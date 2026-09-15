"""Testes de integração das rotas Flask."""


def test_home_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Assistente clínico".encode("utf-8") in resp.data
    assert b"Fase 3" in resp.data
    assert b"Novo exame" not in resp.data
    assert b"/fase1" not in resp.data
    assert b"/fase2" not in resp.data
    assert b"/visao" not in resp.data


def test_health_json(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["fase"] == 3


def test_metrics_json(client):
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "request_count" in data
    assert "workers" in data


def test_paginas_do_assistente_carregam(client):
    for url in (
        "/pacientes",
        "/assistente",
        "/validacao",
        "/alertas",
        "/modelo",
        "/auditoria",
        "/ops/monitoramento",
        "/configuracoes/",
    ):
        assert client.get(url).status_code == 200


def test_modelo_consome_holdout(client):
    resp = client.get("/modelo")
    assert resp.status_code == 200
    assert b"62.8%" in resp.data or "62,8%".encode("utf-8") in resp.data


def test_settings_schema_json(client):
    resp = client.get("/configuracoes/api/schema")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["version"] == 1
    assert any(g["group"] == "aparencia" for g in data["groups"])
    theme_keys = [
        k for g in data["groups"] for k in g["fields"] if k["key"] == "theme"
    ]
    assert theme_keys and {o["value"] for o in theme_keys[0]["options"]} >= {
        "light", "dark", "system"
    }


def test_rota_inexistente_404(client):
    assert client.get("/nao-existe").status_code == 404
