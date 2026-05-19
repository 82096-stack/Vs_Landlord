from __future__ import annotations

import asyncio
import argparse
import json
import os
import re
import webbrowser
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ddz.accounts import AccountManager, DAILY_REPLENISH_LIMIT, RANKED_MIN_RATING, STARTING_RATING
from ddz.connection_manager import ConnectionManager
from ddz.game import GameSession
from ddz.game_room import SeatInfo
from ddz.models import Player
from ddz.pvp import PvpManager
from ddz.rules import MODE_RULES
from ddz.settlement import LOCAL_RANKED_BASE_SCORE, PVP_BASE_SCORE
from ddz.supabase import SupabaseError

# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(title="DouDiZhu WebSocket Server")

manager = ConnectionManager()
accounts = AccountManager()
pvp_manager = PvpManager()


@dataclass
class ClientSession:
    ws: WebSocket
    language: str = "zh"
    username: str = ""
    room_id: str | None = None
    seat: int | None = None
    pvp_room_name: str | None = None


active_sessions: dict[int, ClientSession] = {}
active_users: dict[str, ClientSession] = {}
pvp_live_rooms: dict[str, str] = {}


SUPPORTED_LANGUAGES = {"zh", "en", "hi", "es", "fr", "ar", "bn", "pt", "ru", "ur"}


def msg(type_: str, payload: dict | None = None, request_id: str | None = None) -> dict:
    return {
        "type": type_,
        "room_id": "",
        "timestamp": "",
        "request_id": request_id,
        "payload": payload or {},
    }


async def send_ws(ws: WebSocket, type_: str, payload: dict | None = None, request_id: str | None = None) -> None:
    await ws.send_json(msg(type_, payload, request_id))


def public_stats(username: str) -> dict | None:
    return accounts.get_user_stats(username)


def mode_label_key(mode: str) -> str:
    return "mode_classic" if mode == "classic" else "mode_extended"


def pvp_public_room(room: dict) -> dict:
    return {
        "room_name": room.get("room_name", ""),
        "owner_username": room.get("owner_username", ""),
        "mode": room.get("mode", "classic"),
        "mode_key": mode_label_key(room.get("mode", "classic")),
        "max_rounds": int(room.get("max_rounds", 1)),
        "current_round": int(room.get("current_round", 0)),
        "status": room.get("status", ""),
        "seats": sorted(room.get("seats") or [], key=lambda item: item["seat"]),
        "scores": room.get("scores") or {},
        "winner_username": room.get("winner_username"),
        "has_password": bool(room.get("password")),
    }


def room_name_to_id(room_name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", room_name.strip())[:40]
    return f"PVP_{clean}" if clean else "PVP_ROOM"


def rules_payload() -> dict:
    return {
        "sections": [
            {
                "title_key": "rules_classic_title",
                "body_keys": [
                    "rules_classic_1",
                    "rules_classic_2",
                    "rules_classic_3",
                    "rules_classic_4",
                    "rules_classic_5",
                ],
            },
            {
                "title_key": "rules_extended_title",
                "body_keys": [
                    "rules_extended_1",
                    "rules_extended_2",
                    "rules_extended_3",
                    "rules_extended_4",
                    "rules_extended_5",
                    "rules_extended_6",
                ],
            },
            {
                "title_key": "rules_turn_title",
                "body_keys": [
                    "rules_turn_1",
                    "rules_turn_2",
                    "rules_turn_3",
                    "rules_turn_4",
                    "rules_turn_5",
                ],
            },
            {
                "title_key": "rules_combo_title",
                "body_keys": [
                    "rules_combo_1",
                    "rules_combo_2",
                    "rules_combo_3",
                    "rules_combo_4",
                    "rules_combo_5",
                ],
            },
            {
                "title_key": "rules_scoring_title",
                "body_keys": [
                    "rules_scoring_1",
                    "rules_scoring_2",
                    "rules_scoring_3",
                    "rules_scoring_4",
                    "rules_scoring_5",
                ],
            },
            {
                "title_key": "rules_pvp_title",
                "body_keys": [
                    "rules_pvp_1",
                    "rules_pvp_2",
                    "rules_pvp_3",
                    "rules_pvp_4",
                ],
            },
        ]
    }


async def attach_session_to_room(session: ClientSession, room, seat: int) -> None:
    session.room_id = room.room_id
    session.seat = seat
    room.seats[seat].ws = session.ws
    room.seats[seat].connected = True
    room._start_sender(seat)
    manager.register_ws(session.ws, room.room_id, seat, room.seats[seat].username)


async def start_bound_pvp_round(room_name: str, supabase_room: dict) -> tuple[bool, str, dict | None]:
    live_room_id = pvp_live_rooms.get(room_name)
    if live_room_id:
        old_room = manager.get_room(live_room_id)
        if old_room and old_room.state == "playing":
            return False, "pvp_already_playing", None

    live_room_id = room_name_to_id(room_name)
    suffix = 1
    while live_room_id in manager.rooms:
        existing = manager.rooms[live_room_id]
        if existing.state != "playing":
            manager.remove_room(live_room_id)
            break
        suffix += 1
        live_room_id = f"{room_name_to_id(room_name)}_{suffix}"

    room = manager.create_room(supabase_room["mode"], supabase_room["owner_username"])
    manager.remove_room(room.room_id)
    room.room_id = live_room_id
    room.base_score = PVP_BASE_SCORE
    room.match_kind = "pvp"
    manager.rooms[live_room_id] = room
    pvp_live_rooms[room_name] = live_room_id

    seats = sorted(supabase_room.get("seats") or [], key=lambda item: item["seat"])
    for seat_info in seats:
        seat = room.add_player(seat_info["username"])
        if seat is None:
            return False, "pvp_seat_failed", None
        session = active_users.get(seat_info["username"])
        if session and session.pvp_room_name == room_name:
            await attach_session_to_room(session, room, seat)

    async def finish_round(game_room, players, winner_idx, settlement, deltas):
        landlord = players[game_room.landlord_index].name if game_room.landlord_index is not None else ""
        ok, _message, updated = pvp_manager.record_final_result(
            room_name,
            supabase_room["owner_username"],
            landlord,
            players[winner_idx].role == "landlord",
            max(1, int(settlement["total_score"])),
        )
        payload = {
            "ok": ok,
            "room": pvp_public_room(updated) if updated else None,
            "settlement": settlement,
            "score_deltas": deltas,
            "message_key": "pvp_match_finished" if updated and updated.get("status") == "completed" else "pvp_round_finished",
        }
        await game_room._broadcast("pvp_round_result", payload)
        if updated and updated.get("status") != "completed":
            async def delayed_next_round() -> None:
                await asyncio.sleep(2)
                if game_room.room_id in manager.rooms:
                    await game_room.start_game()
            asyncio.create_task(delayed_next_round())

    room.round_finished_callback = finish_round
    await room.start_game()
    return True, "pvp_started", pvp_public_room(supabase_room)


async def disband_pvp_room_for_all(room_name: str, actor_username: str) -> tuple[bool, str]:
    ok, message = pvp_manager.disband_room(actor_username, room_name)
    if not ok:
        return ok, message

    live_id = pvp_live_rooms.pop(room_name, None)
    live_room = manager.get_room(live_id) if live_id else None
    affected: list[ClientSession] = [
        session
        for session in list(active_sessions.values())
        if session.pvp_room_name == room_name
    ]
    if live_room is not None:
        await live_room._broadcast("pvp_room_disbanded", {
            "room_name": room_name,
            "message": message,
            "message_key": "room_disbanded",
        })
        manager.remove_room(live_room.room_id)

    for session in affected:
        session.pvp_room_name = None
        if session.room_id == live_id:
            session.room_id = None
            session.seat = None
        try:
            await send_ws(session.ws, "pvp_room_disbanded", {
                "room_name": room_name,
                "message": message,
                "message_key": "room_disbanded",
            })
        except Exception:
            pass
    return ok, message


async def start_local_ai_game(session: ClientSession, mode: str, match_type: str, request_id: str | None) -> None:
    if not session.username:
        await send_ws(session.ws, "error", {"message_key": "login_required"}, request_id)
        return
    if mode not in MODE_RULES:
        mode = "classic"
    if match_type == "ranked":
        ok, _message, stats = accounts.prepare_ranked_entry(session.username)
        await send_ws(session.ws, "ranked_entry", {
            "ok": ok,
            "message_key": "ranked_entry_ok" if ok else "ranked_entry_denied",
            "stats": stats,
        }, request_id)
        if not ok:
            return

    room = manager.create_room(mode, session.username)
    seat = room.add_player(session.username)
    if seat is None:
        await send_ws(session.ws, "error", {"message_key": "room_join_failed"}, request_id)
        return
    room.base_score = LOCAL_RANKED_BASE_SCORE
    room.match_kind = "local_ranked" if match_type == "ranked" else "casual_no_score"
    await attach_session_to_room(session, room, seat)
    room.fill_with_ai()

    async def finish_local(game_room, players, winner_idx, settlement, deltas):
        human = players[seat]
        won = human.role == players[winner_idx].role
        delta = int(deltas.get(human.name, 0)) if match_type == "ranked" else 0
        accounts.record_result(session.username, won, match_type=match_type, rating_delta=delta)
        await game_room._send_to(seat, "stats_updated", {
            "stats": public_stats(session.username),
            "rating_delta": delta,
        })

    room.round_finished_callback = finish_local
    await send_ws(session.ws, "local_game_created", {
        "room_id": room.room_id,
        "seat": seat,
        "mode": mode,
        "match_type": match_type,
    }, request_id)
    await room._broadcast("room_state", room.public_room_state()["payload"])
    await room.start_game()

# ============================================================
# HTML Frontend
# ============================================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>斗地主</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0d1b2a;color:#e0e1dd;min-height:100vh}
.container{max-width:900px;margin:0 auto;padding:20px}
h1{text-align:center;color:#f4a261;margin:20px 0}
.panel{background:#1b2838;border:1px solid #2a3a4a;border-radius:12px;padding:16px;margin:10px 0}
.panel h3{color:#e9c46a;margin-bottom:10px}
input,button{font-size:1rem;padding:8px 12px;border-radius:6px;border:1px solid #3a4a5a;background:#16212e;color:#e0e1dd;margin:4px}
button{background:#f4a261;color:#0d1b2a;font-weight:bold;cursor:pointer}
button:hover{background:#e76f51}
button:disabled{background:#555;cursor:not-allowed}
.row{display:flex;gap:6px;flex-wrap:wrap}
.row>*{flex:1;min-width:120px}
#log{background:#0a1118;border:1px solid #2a3a4a;border-radius:8px;padding:12px;height:280px;overflow-y:auto;font-family:monospace;font-size:.85rem;white-space:pre-wrap}
#hand{min-height:50px;background:#0a1118;border:1px solid #2a3a4a;border-radius:8px;padding:10px;font-family:monospace}
.card-btn{display:inline-block;background:#f4a261;color:#0d1b2a;padding:5px 8px;margin:3px;border-radius:5px;cursor:pointer;font-weight:bold;font-size:.85rem;border:2px solid transparent;user-select:none}
.card-btn.selected{border-color:#e76f51;background:#e76f51;color:#fff}
</style>
</head>
<body>
<div class="container">
<h1>斗地主 WebSocket</h1>
<div class="panel">
<h3>连接</h3>
<div class="row">
<input id="username" placeholder="昵称" maxlength="20">
<button onclick="createRoom()">创建房间</button>
<input id="joinRoomId" placeholder="房间号" maxlength="6">
<button onclick="joinRoom()">加入房间</button>
</div>
<button id="startBtn" onclick="startGame()" style="display:none;background:#2ecc71">开始游戏</button>
<div id="roomInfo" style="margin-top:8px;color:#e9c46a"></div>
</div>
<div class="panel" id="gamePanel" style="display:none">
<h3>游戏面板</h3>
<div id="gameState" style="color:#9aa5b1;margin-bottom:8px"></div>
<div id="handContainer">
<label>你的手牌 <span id="cardCount">0</span> 张</label>
<div id="hand"></div>
</div>
<div class="row" style="margin-top:8px">
<button id="playBtn" onclick="playCards()" disabled>出牌</button>
<button id="passBtn" onclick="send({type:'pass'})" disabled>过牌</button>
</div>
<div id="bidPanel" style="display:none;margin-top:8px">
<label>叫地主</label>
<div class="row" id="bidButtons"></div>
</div>
</div>
<div class="panel">
<h3>日志</h3>
<div id="log"></div>
</div>
</div>
<script>
let ws=null,roomId=null,mySeat=null,myCards=[],selected=new Set(),reqCounter=0;
function rid(){return 'r'+Date.now()+'_'+(++reqCounter)}
function L(m){const e=document.getElementById('log');const t=new Date().toLocaleTimeString();e.textContent+='['+t+'] '+m+'\\n';e.scrollTop=e.scrollHeight}
function connect(){const p=location.protocol==='https:'?'wss:':'ws:';ws=new WebSocket(p+'//'+location.host+'/ws');ws.onopen=()=>L('已连接');ws.onclose=()=>L('已断开');ws.onmessage=e=>{handle(JSON.parse(e.data))}}
function send(d){if(ws&&ws.readyState===WebSocket.OPEN){d.request_id=d.request_id||rid();ws.send(JSON.stringify(d))}}
function createRoom(){const u=document.getElementById('username').value.trim();if(!u){L('请输入昵称');return}send({type:'create_room',username:u,mode:'classic'})}
function joinRoom(){const u=document.getElementById('username').value.trim();const rid=document.getElementById('joinRoomId').value.trim().toUpperCase();if(!u||!rid){L('请输入昵称和房间号');return}send({type:'join_room',room_id:rid,username:u})}
function startGame(){send({type:'start_game'})}
function playCards(){if(selected.size===0)return;const cards=Array.from(selected).map(i=>myCards[i]);const idx=Array.from(selected).sort((a,b)=>a-b);send({type:'play_card',cards:idx,action:'play'});
selected.clear();document.querySelectorAll('.card-btn.selected').forEach(b=>b.classList.remove('selected'))}
function handle(msg){
    const type=msg.type;
    const p=msg.payload||{};
    const rid=msg.request_id||'';
    L('收到: '+type+(rid?' ['+rid+']':''));
    switch(type){
    case'room_created':roomId=p.room_id||msg.room_id;document.getElementById('roomInfo').innerHTML='房间 <b>'+roomId+'</b> 已创建';document.getElementById('gamePanel').style.display='block';break;
    case'room_joined':roomId=p.room_id||msg.room_id;mySeat=p.seat!=null?p.seat:msg.seat;document.getElementById('roomInfo').innerHTML='已加入 <b>'+roomId+'</b> 座位'+(mySeat+1);document.getElementById('gamePanel').style.display='block';break;
    case'room_state':updateRoom(p);break;
    case'game_starting':L('游戏开始! '+((p.mode_label)||''));document.getElementById('gameState').textContent='游戏中';break;
    case'game_started':L('游戏开始!');document.getElementById('gameState').textContent='游戏中';break;
    case'your_hand':case'your_cards':myCards=p.cards||[];renderHand();document.getElementById('cardCount').textContent=myCards.length;break;
    case'cards_dealt':L('标记牌: '+p.marked_card+' 底牌: '+p.bottom_count+'张');break;
    case'ask_bid':showBid(p.allowed_bids||[0,1,2,3]);break;
    case'ask_call':showCall();break;
    case'ask_rob':showRob(p.highest_bid||0);break;
    case'bid_result':L('座位'+p.seat+' '+p.player_name+': '+(p.bid===0?'不叫':p.bid+'分'));break;
    case'call_result':L('座位'+p.seat+' '+p.player_name+': '+(p.call?'叫地主':'不叫'));break;
    case'rob_result':L('座位'+p.seat+' '+p.player_name+': '+(p.rob?'抢地主':'不抢'));break;
    case'bidding_turn':break;
    case'no_bidder':L(p.message||'无人叫地主');break;
    case'redeal':L(p.message||'重新发牌');break;
    case'landlord_assigned':L('座位'+p.seat+' '+p.player_name+' 成为地主，底牌: '+(p.bottom_cards||[]).map(c=>c.label).join(' '));renderHand();break;
    case'play_turn':L('轮到 座位'+p.seat+' '+p.player_name+(p.is_opening?' (新回合)':''));break;
    case'ask_play':L(p.is_opening?'请出牌（新回合）':'请出牌，压过 '+((p.last_combo&&p.last_combo.description)||'?'));document.getElementById('playBtn').disabled=false;document.getElementById('passBtn').disabled=p.is_opening;renderHand();break;
    case'play_action':
        if(p.action==='play'){
            L('座位'+p.seat+' '+p.player_name+' 出牌 '+p.combo_display+' 剩'+p.remaining_count+'张');
        }else{
            L('座位'+p.seat+' '+p.player_name+' 过牌');
        }
        document.getElementById('playBtn').disabled=true;document.getElementById('passBtn').disabled=true;
        break;
    case'player_empty':L('座位'+p.seat+' '+p.player_name+' 手牌出完!');break;
    case'player_disconnected':L('座位'+p.seat+' '+p.username+' 断开连接');break;
    case'new_round':L('新回合: '+p.leader_name);break;
    case'reveal_result':L('座位'+p.seat+' '+p.player_name+(p.reveal?' 摊打':' 不摊打'));break;
    case'report_result':L('座位'+p.seat+' '+p.player_name+' '+p.report_label);break;
    case'game_over':L('游戏结束! 胜者: '+p.winner_name+' ('+p.winner_role+') 地主: '+p.landlord_name);document.getElementById('playBtn').disabled=true;document.getElementById('passBtn').disabled=true;break;
    case'state_snapshot':L('收到状态快照 (重连恢复)');if(p.players)updateRoomFromSnapshot(p);break;
    case'error':L('错误: '+(p.message||msg.message||''));break;
    default:L('未知消息: '+type);
    }
}
function updateRoom(p){
    const ps=(p.players||[]).map(function(x){return '座位'+x.seat+': '+(x.username||'空')+' '+(x.role==='landlord'?'[地主]':'')+' '+x.hand_size+'张'}).join(' | ');
    L('房间 '+(p.room_id||roomId)+' '+(p.state||'')+'\\n'+ps);
    if(p.state==='waiting'&&roomId){
        var filled=(p.players||[]).filter(function(x){return x.username}).length>=3;
        document.getElementById('startBtn').style.display=filled?'inline-block':'none';
    }
}
function updateRoomFromSnapshot(p){
    L('房间恢复: '+(p.room_id||roomId)+' '+p.state);
    if(p.landlord_index!=null)L('地主: 座位'+p.landlord_index);
    if(p.current_turn!=null)L('当前回合: 座位'+p.current_turn);
}
function renderHand(){var c=document.getElementById('hand');selected.clear();c.innerHTML=myCards.map(function(s,i){return '<span class="card-btn" onclick="toggle('+i+')" id="c-'+i+'">'+(s.label||s)+'</span>'}).join('')}
function toggle(i){var b=document.getElementById('c-'+i);if(selected.has(i)){selected.delete(i);b.classList.remove('selected')}else{selected.add(i);b.classList.add('selected')}}
function showBid(allowed){var p=document.getElementById('bidPanel');var btns=document.getElementById('bidButtons');p.style.display='block';btns.innerHTML=allowed.map(function(b){return '<button onclick="send({type:\\'bid\\',bid:'+b+'});document.getElementById(\\'bidPanel\\').style.display=\\'none\\'">'+(b===0?'不叫':b+'分')+'</button>'}).join('')}
function showCall(){var p=document.getElementById('bidPanel');var btns=document.getElementById('bidButtons');p.style.display='block';btns.innerHTML='<button onclick="send({type:\\'call\\',call:true});document.getElementById(\\'bidPanel\\').style.display=\\'none\\'">叫地主</button><button onclick="send({type:\\'call\\',call:false});document.getElementById(\\'bidPanel\\').style.display=\\'none\\'">不叫</button>'}
function showRob(bid){var p=document.getElementById('bidPanel');var btns=document.getElementById('bidButtons');p.style.display='block';btns.innerHTML='<button onclick="send({type:\\'rob\\',rob:true});document.getElementById(\\'bidPanel\\').style.display=\\'none\\'">抢地主</button><button onclick="send({type:\\'rob\\',rob:false});document.getElementById(\\'bidPanel\\').style.display=\\'none\\'">不抢</button>'}
connect();
</script>
</body>
</html>"""

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dou Dizhu</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#111827;color:#f3f4f6;min-height:100vh}button,input,select{font:inherit}button{border:0;border-radius:6px;background:#f59e0b;color:#111827;font-weight:700;padding:10px 12px;cursor:pointer}button:hover{background:#f97316}button:disabled{background:#4b5563;color:#9ca3af;cursor:not-allowed}input,select{width:100%;border:1px solid #374151;border-radius:6px;background:#0f172a;color:#f9fafb;padding:10px}.app{max-width:1180px;margin:0 auto;padding:18px}.top{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:14px}.brand{font-size:1.6rem;font-weight:800;color:#fbbf24}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.panel{background:#1f2937;border:1px solid #374151;border-radius:8px;padding:14px}.panel h2,.panel h3{margin:0 0 10px;color:#fde68a}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:end}.row>*{flex:1;min-width:140px}.hidden{display:none!important}.muted{color:#9ca3af}.danger{background:#ef4444;color:#fff}.ok{background:#22c55e;color:#052e16}.secondary{background:#334155;color:#f8fafc}.cards{min-height:74px;background:#0f172a;border:1px solid #334155;border-radius:8px;padding:8px}.card{display:inline-block;margin:3px;padding:7px 9px;border-radius:6px;background:#f8fafc;color:#111827;font-weight:800;cursor:pointer;border:2px solid transparent}.card.selected{background:#fb923c;color:#111827;border-color:#fed7aa}.log{height:260px;overflow:auto;white-space:pre-wrap;background:#020617;border:1px solid #334155;border-radius:8px;padding:10px;font-family:ui-monospace,Menlo,monospace;font-size:.86rem}.stack{display:flex;flex-direction:column;gap:8px}.pill{display:inline-block;border:1px solid #475569;border-radius:999px;padding:4px 9px;margin:2px;color:#e2e8f0}.password-hint{font-size:.82rem;color:#cbd5e1;margin:4px 0}.toolbar{display:flex;gap:8px;flex-wrap:wrap}.lang-menu{min-width:170px}.room-list{display:grid;gap:8px}.room-card{border:1px solid #475569;border-radius:8px;padding:10px;background:#111827}.wide{grid-column:1/-1}a{color:#93c5fd}
</style>
</head>
<body>
<div class="app">
  <div class="top">
    <div class="brand" data-i18n="app_title"></div>
    <div class="toolbar">
      <select id="languageSelect" class="lang-menu" onchange="setLanguage(this.value)"></select>
      <button onclick="showView(state.username?'home':'auth')">language</button>
    </div>
  </div>
  <div id="status" class="panel muted"></div>

  <section id="authView" class="grid">
    <div class="panel stack">
      <h2 data-i18n="login"></h2>
      <input id="loginUser" data-i18n-placeholder="username" maxlength="20">
      <div class="password-hint" data-i18n="password_hint"></div>
      <input id="loginPass" class="password-input" type="password" data-i18n-placeholder="password">
      <button onclick="login()" data-i18n="login"></button>
    </div>
    <div class="panel stack">
      <h2 data-i18n="register"></h2>
      <input id="registerUser" data-i18n-placeholder="username" maxlength="20">
      <div class="password-hint" data-i18n="password_hint"></div>
      <input id="registerPass" class="password-input" type="password" data-i18n-placeholder="password">
      <button onclick="registerAccount()" data-i18n="register"></button>
    </div>
    <div class="panel stack">
      <h2 data-i18n="ai_demo"></h2>
      <select id="demoMode"></select>
      <input id="demoRounds" type="number" value="1" min="1" max="50">
      <button onclick="startAiDemo()" data-i18n="start_demo"></button>
    </div>
  </section>

  <section id="homeView" class="grid hidden">
    <div class="panel wide">
      <h2><span data-i18n="welcome"></span> <span id="who"></span></h2>
      <div class="toolbar">
        <button onclick="showView('local')" data-i18n="local_ai"></button>
        <button onclick="send({type:'get_stats'})" data-i18n="stats"></button>
        <button onclick="showView('pvp');send({type:'pvp_list_rooms'})" data-i18n="online_pvp"></button>
        <button onclick="send({type:'get_rules'})" data-i18n="rules"></button>
        <button class="danger" onclick="send({type:'logout'})" data-i18n="logout"></button>
      </div>
    </div>
  </section>

  <section id="localView" class="grid hidden">
    <div class="panel stack">
      <h2 data-i18n="local_ai"></h2>
      <select id="localMode"></select>
      <select id="localMatch"></select>
      <button onclick="startLocal()" data-i18n="start_game"></button>
      <button class="secondary" onclick="showView('home')" data-i18n="back"></button>
    </div>
  </section>

  <section id="pvpView" class="grid hidden">
    <div class="panel stack">
      <h2 data-i18n="create_room"></h2>
      <input id="pvpRoomName" data-i18n-placeholder="room_name">
      <div class="password-hint" data-i18n="password_hint"></div>
      <input id="pvpRoomPass" class="password-input" type="password" data-i18n-placeholder="room_password_optional">
      <select id="pvpMode"></select>
      <input id="pvpRounds" type="number" value="1" min="1" max="50">
      <button onclick="createPvp()" data-i18n="create_room"></button>
    </div>
    <div class="panel stack">
      <h2 data-i18n="join_room"></h2>
      <input id="joinPvpName" data-i18n-placeholder="room_name">
      <div class="password-hint" data-i18n="password_hint"></div>
      <input id="joinPvpPass" class="password-input" type="password" data-i18n-placeholder="room_password_optional">
      <button onclick="joinPvp()" data-i18n="join_room"></button>
      <button onclick="send({type:'pvp_list_rooms'})" data-i18n="refresh"></button>
      <button class="secondary" onclick="showView('home')" data-i18n="back"></button>
    </div>
    <div class="panel wide">
      <h2 data-i18n="rooms"></h2>
      <div id="rooms" class="room-list"></div>
    </div>
  </section>

  <section id="rulesView" class="grid hidden">
    <div class="panel wide">
      <h2 data-i18n="rules"></h2>
      <div id="rulesContent"></div>
      <button class="secondary" onclick="showView(state.username?'home':'auth')" data-i18n="back"></button>
    </div>
  </section>

  <section id="gameView" class="grid hidden">
    <div class="panel wide">
      <h2 data-i18n="game_table"></h2>
      <div id="gameInfo" class="muted"></div>
      <div id="players"></div>
      <div class="toolbar" style="margin-top:8px">
        <button id="startRoomBtn" class="ok hidden" onclick="send({type:'start_game'})" data-i18n="start_game"></button>
        <button id="startPvpBtn" class="ok hidden" onclick="send({type:'pvp_start_match',room_name:state.pvpRoom})" data-i18n="start_match"></button>
        <button class="secondary" onclick="showView(state.username?'home':'auth')" data-i18n="back"></button>
      </div>
    </div>
    <div class="panel wide">
      <h3><span data-i18n="your_hand"></span> <span id="cardCount">0</span></h3>
      <div id="hand" class="cards"></div>
      <div id="actionPanel" class="toolbar" style="margin-top:8px"></div>
    </div>
  </section>

  <section class="grid">
    <div class="panel wide">
      <h2 data-i18n="log"></h2>
      <div id="log" class="log"></div>
    </div>
  </section>
</div>
<script>
const LANGS=[
 ['zh','汉语'],['en','English'],['hi','हिन्दी'],['es','Español'],['fr','Français'],['ar','العربية'],['bn','বাংলা'],['pt','Português'],['ru','Русский'],['ur','اردو']
];
const BASE={
 app_title:'Dou Dizhu WebSocket',login:'Login',register:'Register',username:'Username',password:'Password',password_hint:'Press Control+P while this field is focused to show or hide the password.',ai_demo:'AI match',start_demo:'Start demo',welcome:'Welcome',local_ai:'Local AI match',stats:'Stats',online_pvp:'Online PVP',rules:'Rules',logout:'Logout',start_game:'Start game',back:'Back',create_room:'Create room',join_room:'Join room',disband_room:'Disband room',room_name:'Room name',room_password_optional:'Room password (optional)',refresh:'Refresh',rooms:'Rooms',game_table:'Game table',your_hand:'Your hand',log:'Log',mode_classic:'Two vs one',mode_extended:'Three vs one',ranked:'Ranked',casual:'Casual',start_match:'Start match',seat:'Seat',empty:'Empty',landlord:'Landlord',farmer:'Farmer',connected:'Connected',disconnected:'Disconnected',waiting:'Waiting',playing:'Playing',finished:'Finished',completed:'Completed',lobby:'Lobby',bid:'Bid',no_bid:'No bid',call_landlord:'Call landlord',dont_call:'Do not call',rob_landlord:'Rob landlord',dont_rob:'Do not rob',reveal:'Reveal',dont_reveal:'Do not reveal',report:'Report',dont_report:'Do not report',play:'Play',pass:'Pass',round:'Round',score:'Score',winner:'Winner',base_score:'Base score',multiplier:'Multiplier',bomb:'Bomb/Rocket',reveal_factor:'Reveal',redeal_factor:'Redeal',report_factor:'Report',marker_factor:'Joker marker',marker_card:'Marked card',marker_holder:'Marked-card holder',bottom_cards:'Bottom cards',spring:'Spring',reverse_spring:'Reverse spring',total:'Total',small_joker:'Small Joker',big_joker:'Big Joker',single:'Single',pair:'Pair',trio:'Trio',trio_single:'Trio with single',trio_pair:'Trio with pair',straight:'Straight',pair_straight:'Pair straight',trio_straight:'Plane',airplane_single:'Plane with singles',airplane_pair:'Plane with pairs',four_two_single:'Four with two',four_two_pair:'Four with two pairs',rocket:'Rocket',room_created:'Room created.',room_joined:'Joined room.',room_disbanded:'Room disbanded. All players were returned to the lobby.',login_success:'Login successful.',register_success:'Registration successful.',logged_out:'Logged out.',login_required:'Please log in first.',stats_updated:'Stats updated.',pvp_started:'PVP match started.',pvp_round_finished:'PVP round finished.',pvp_match_finished:'PVP match finished.',ranked_entry_ok:'Ranked entry accepted.',ranked_entry_denied:'Ranked entry denied.',room_join_failed:'Could not join room.',pvp_already_playing:'This PVP match is already playing.',pvp_seat_failed:'Could not create PVP seats.',error:'Error',rules_classic_title:'Goal: two farmers vs one landlord',rules_classic_1:'Classic mode has 3 players. One player becomes the landlord, and the other two players are farmers on the same team.',rules_classic_2:'The deck has 54 cards. Each player receives 17 cards, and 3 bottom cards are left face down for the future landlord.',rules_classic_3:'The player who receives the marked card starts the landlord decision. A player may call landlord, then later players may rob landlord.',rules_classic_4:'After bidding, the landlord takes the 3 bottom cards into their hand and plays first.',rules_classic_5:'The landlord wins by emptying their hand first. The farmers win if either farmer empties their hand first.',rules_extended_title:'Three vs one mode',rules_extended_1:'Extended mode has 4 players. One landlord fights three farmers.',rules_extended_2:'It uses two decks. Each player receives 25 cards, and the landlord receives 8 bottom cards.',rules_extended_3:'The landlord may reveal their hand. Revealing increases risk but also increases the score multiplier.',rules_extended_4:'Some hands may report. Report and double report add extra multipliers.',rules_extended_5:'Farmers have bomb limits based on bidding. A farmer who bid higher can usually use more bombs.',rules_extended_6:'Straights are more flexible in this mode, including A2345 style sequences.',rules_turn_title:'How a turn works',rules_turn_1:'The first player in a trick may play any legal combination from their hand.',rules_turn_2:'The next player must play the same kind of combination with a higher main rank, or pass.',rules_turn_3:'Bombs can beat normal combinations. Rockets beat bombs and all normal combinations.',rules_turn_4:'If everyone else passes, the last player who played cards starts a new trick and may choose any legal combination.',rules_turn_5:'Keep selecting cards from your hand, then press Play. If you cannot or do not want to beat the table, press Pass.',rules_combo_title:'Common card combinations',rules_combo_1:'Single: one card. Pair: two cards of the same rank. Trio: three cards of the same rank.',rules_combo_2:'Trio with single or pair: three of a kind plus one extra card or one pair.',rules_combo_3:'Straight: at least five consecutive single cards. Pair straight: at least three consecutive pairs.',rules_combo_4:'Plane: consecutive trios, optionally with matching extra singles or pairs.',rules_combo_5:'Bomb: four or more cards of the same rank. Rocket: jokers together, the strongest combination.',rules_scoring_title:'Scoring',rules_scoring_1:'Ranked local matches start at 1200 points and require at least 200 points to enter.',rules_scoring_2:'If your rating is below 200, the game can refill you to 1200 points up to twice per day.',rules_scoring_3:'Ranked base score is 50. PVP base score is 1.',rules_scoring_4:'The bid is the base multiplier. Bombs, rockets, reveal, redeals, reports, spring, reverse spring and joker marker can double the score again.',rules_scoring_5:'Casual matches record wins and losses but do not change rating.',rules_pvp_title:'Online PVP',rules_pvp_1:'The owner creates a room, optionally sets a password, chooses two-vs-one or three-vs-one, and chooses the number of rounds.',rules_pvp_2:'Other players search the room list, enter the password if needed, and join the room.',rules_pvp_3:'When seats are full, the owner starts the match. Players then play in turn through the web interface.',rules_pvp_4:'After each round, scores are calculated automatically. When all rounds finish, the highest total score wins.'};
const OV={
 zh:{app_title:'斗地主 WebSocket',login:'登录账号',register:'注册账号',username:'用户名',password:'密码',password_hint:'聚焦此密码框时按 Control+P 可显示或隐藏密码。',ai_demo:'AI 对战',start_demo:'开始演示',welcome:'欢迎',local_ai:'本地 AI 对战',stats:'查看战绩',online_pvp:'线上 PVP',rules:'阅读规则',logout:'退出登录',start_game:'开始游戏',back:'返回',create_room:'创建房间',join_room:'加入房间',disband_room:'解散房间',room_name:'房间名',room_password_optional:'房间密码（可空）',refresh:'刷新',rooms:'房间列表',game_table:'游戏桌',your_hand:'你的手牌',log:'日志',mode_classic:'二打一',mode_extended:'三打一',ranked:'积分赛',casual:'娱乐赛',start_match:'开始比赛',seat:'座位',empty:'空',landlord:'地主',farmer:'农民',connected:'在线',disconnected:'离线',waiting:'等待中',playing:'游戏中',finished:'已结束',completed:'已完成',lobby:'大厅',bid:'叫分',no_bid:'不叫',call_landlord:'叫地主',dont_call:'不叫',rob_landlord:'抢地主',dont_rob:'不抢',reveal:'摊打',dont_reveal:'不摊打',report:'报道',dont_report:'不报道',play:'出牌',pass:'过牌',round:'轮',score:'分数',winner:'胜利者',base_score:'基本分',multiplier:'倍率',bomb:'炸弹/王炸',reveal_factor:'摊打',redeal_factor:'荒番',report_factor:'报道',marker_factor:'标记王',marker_card:'标记牌',marker_holder:'标记牌玩家',bottom_cards:'底牌',spring:'春天',reverse_spring:'反春天',total:'总分',small_joker:'小王',big_joker:'大王',single:'单张',pair:'对子',trio:'三张',trio_single:'三带一',trio_pair:'三带二',straight:'顺子',pair_straight:'连对',trio_straight:'飞机不带',airplane_single:'飞机带单',airplane_pair:'飞机带对',four_two_single:'四带二',four_two_pair:'四带两对',rocket:'王炸',room_created:'房间已创建。',room_joined:'已加入房间。',room_disbanded:'房间已解散，所有玩家已回到大厅。',login_success:'登录成功。',register_success:'注册成功。',logged_out:'已退出登录。',login_required:'请先登录。',stats_updated:'战绩已更新。',pvp_started:'PVP 比赛已开始。',pvp_round_finished:'PVP 本轮已结算。',pvp_match_finished:'PVP 比赛已结束。',ranked_entry_ok:'可以进入积分赛。',ranked_entry_denied:'不能进入积分赛。',room_join_failed:'加入房间失败。',pvp_already_playing:'该 PVP 正在比赛中。',pvp_seat_failed:'创建 PVP 座位失败。',error:'错误',rules_classic_title:'目标：二打一',rules_classic_1:'经典模式共有 3 名玩家。1 人是地主，另外 2 人是农民，农民属于同一队。',rules_classic_2:'使用 54 张牌。每人先拿 17 张，剩下 3 张作为底牌，之后交给地主。',rules_classic_3:'拿到标记牌的玩家先决定是否叫地主。有人叫地主后，后面的玩家可以选择抢地主。',rules_classic_4:'叫抢结束后，最终地主拿走 3 张底牌，并由地主先出牌。',rules_classic_5:'地主先出完手牌则地主胜；任意农民先出完手牌则农民队胜。',rules_extended_title:'三打一模式',rules_extended_1:'三打一共有 4 名玩家。1 人是地主，3 人是农民。',rules_extended_2:'使用两副牌。每人 25 张，地主获得 8 张底牌。',rules_extended_3:'地主可以选择摊打，也就是亮出手牌；这样风险更大，但会增加倍数。',rules_extended_4:'部分强手牌可以报道或双报道，报道会继续增加倍数。',rules_extended_5:'农民能用炸弹的次数和叫分有关，叫得越高通常可用炸弹次数越多。',rules_extended_6:'三打一中的顺子规则更宽，允许 A2345 这样的顺子。',rules_turn_title:'一轮出牌怎么进行',rules_turn_1:'一轮开始时，领出玩家可以出任意合法牌型。',rules_turn_2:'后面的玩家必须出同牌型且更大的牌，或者选择过牌。',rules_turn_3:'炸弹可以压普通牌型；王炸是最强牌型，可以压过炸弹和普通牌型。',rules_turn_4:'如果其他玩家都过牌，最后成功出牌的玩家重新领出，开启新一轮。',rules_turn_5:'在网页里点击手牌选牌，再按出牌；如果不想出或压不过，就按过牌。',rules_combo_title:'常见牌型',rules_combo_1:'单张是一张牌；对子是两张同点数牌；三张是三张同点数牌。',rules_combo_2:'三带一/三带二是三张同点数牌，再带一张单牌或一个对子。',rules_combo_3:'顺子是至少五张连续单牌；连对是至少三组连续对子。',rules_combo_4:'飞机是连续的三张组合，可以不带，也可以带单牌或对子。',rules_combo_5:'炸弹是四张或更多同点数牌；王炸由王牌组成，是最强牌。',rules_scoring_title:'计分',rules_scoring_1:'本地积分赛初始 1200 分，至少 200 分才能参加。',rules_scoring_2:'低于 200 分时，每天最多可以自动补分到 1200 分两次。',rules_scoring_3:'本地积分赛基本分是 50，线上 PVP 基本分是 1。',rules_scoring_4:'叫分是基础倍数。炸弹/王炸、摊打、荒番、报道、春天、反春天和标记王都会继续翻倍。',rules_scoring_5:'娱乐赛只记录胜负，不改变积分。',rules_pvp_title:'线上 PVP',rules_pvp_1:'房主创建房间，可以设置密码、玩法和比赛轮数。',rules_pvp_2:'其他玩家在房间列表中找到房间，有密码则输入密码后加入。',rules_pvp_3:'座位满后，房主点击开始比赛，所有玩家在网页中按顺序出牌。',rules_pvp_4:'每轮结束后系统自动结算分数；所有轮数结束后，总分最高者获胜。'},
 es:{login:'Iniciar sesión',register:'Registrarse',ai_demo:'Partida IA',local_ai:'IA local',stats:'Estadísticas',online_pvp:'PVP en línea',rules:'Reglas',logout:'Salir',password_hint:'Pulsa Control+P en este campo para mostrar u ocultar la contraseña.'},
 fr:{login:'Connexion',register:'Créer un compte',ai_demo:'Partie IA',local_ai:'IA locale',stats:'Statistiques',online_pvp:'PVP en ligne',rules:'Règles',logout:'Déconnexion',password_hint:'Appuyez sur Control+P dans ce champ pour afficher ou masquer le mot de passe.'},
 pt:{login:'Entrar',register:'Registrar',ai_demo:'Jogo IA',local_ai:'IA local',stats:'Estatísticas',online_pvp:'PVP online',rules:'Regras',logout:'Sair',password_hint:'Pressione Control+P neste campo para mostrar ou ocultar a senha.'},
 ru:{login:'Вход',register:'Регистрация',ai_demo:'Игра ИИ',local_ai:'Локальная игра с ИИ',stats:'Статистика',online_pvp:'Онлайн PVP',rules:'Правила',logout:'Выйти',password_hint:'Нажмите Control+P в этом поле, чтобы показать или скрыть пароль.'},
 hi:{login:'लॉगिन',register:'रजिस्टर',ai_demo:'AI मैच',local_ai:'स्थानीय AI मैच',stats:'आँकड़े',online_pvp:'ऑनलाइन PVP',rules:'नियम',logout:'लॉग आउट',password_hint:'पासवर्ड दिखाने या छिपाने के लिए इस फ़ील्ड पर Control+P दबाएँ।'},
 bn:{login:'লগইন',register:'নিবন্ধন',ai_demo:'AI ম্যাচ',local_ai:'স্থানীয় AI ম্যাচ',stats:'পরিসংখ্যান',online_pvp:'অনলাইন PVP',rules:'নিয়ম',logout:'লগআউট',password_hint:'পাসওয়ার্ড দেখাতে বা লুকাতে এই ঘরে Control+P চাপুন।'},
 ar:{login:'تسجيل الدخول',register:'تسجيل حساب',ai_demo:'مباراة ذكاء اصطناعي',local_ai:'لعب محلي ضد الذكاء',stats:'الإحصاءات',online_pvp:'PVP عبر الإنترنت',rules:'القواعد',logout:'خروج',password_hint:'اضغط Control+P داخل هذا الحقل لإظهار كلمة المرور أو إخفائها.'},
 ur:{login:'لاگ اِن',register:'رجسٹر',ai_demo:'AI میچ',local_ai:'مقامی AI میچ',stats:'اعدادوشمار',online_pvp:'آن لائن PVP',rules:'قواعد',logout:'لاگ آؤٹ',password_hint:'پاس ورڈ دکھانے یا چھپانے کے لیے اس خانے میں Control+P دبائیں۔'}
};
const state={lang:'zh',username:'',ws:null,cards:[],selected:new Set(),seat:null,roomId:null,pvpRoom:null};
function tr(k,p={}){let s=(OV[state.lang]&&OV[state.lang][k])||BASE[k]||k;return s.replace(/\\{(\\w+)\\}/g,(_,x)=>p[x]??'')}
function localizeCard(c){const label=c.label||c;if(label==='小王')return tr('small_joker');if(label==='大王')return tr('big_joker');return label}
function localizeCombo(c){if(!c)return '';return tr(c.kind||c.display_name||'')+(c.sequence_length>1?' '+c.sequence_length:'')}
function send(d){if(state.ws&&state.ws.readyState===1){d.request_id='r'+Date.now()+Math.random();state.ws.send(JSON.stringify(d))}}
function logLine(k,p={}){const e=document.getElementById('log');e.textContent+='['+new Date().toLocaleTimeString()+'] '+tr(k,p)+'\\n';e.scrollTop=e.scrollHeight}
function logText(text){const e=document.getElementById('log');e.textContent+='['+new Date().toLocaleTimeString()+'] '+text+'\\n';e.scrollTop=e.scrollHeight}
function reason(p){return p.message || (p.message_key?tr(p.message_key,p.params||{}):tr('error'))}
function cardsText(cards){return (cards||[]).map(localizeCard).join(' ')}
function comboText(combo, fallback){return combo?localizeCombo(combo):(fallback||'')}
function roleText(role){return tr(role==='landlord'?'landlord':'farmer')}
function yesNoText(ok, yesKey, noKey){return ok?tr(yesKey):tr(noKey)}
function showView(v){for(const id of ['authView','homeView','localView','pvpView','rulesView','gameView'])document.getElementById(id).classList.add('hidden');document.getElementById(v+'View').classList.remove('hidden')}
function applyI18n(){document.documentElement.lang=state.lang;document.documentElement.dir=['ar','ur'].includes(state.lang)?'rtl':'ltr';document.querySelectorAll('[data-i18n]').forEach(e=>e.textContent=tr(e.dataset.i18n));document.querySelectorAll('[data-i18n-placeholder]').forEach(e=>e.placeholder=tr(e.dataset.i18nPlaceholder));fillSelects();renderHand();renderStatus()}
function fillSelects(){const lang=document.getElementById('languageSelect');if(!lang.dataset.ready){lang.innerHTML=LANGS.map(x=>`<option value="${x[0]}">${x[1]}</option>`).join('');lang.dataset.ready=1}lang.value=state.lang;for(const id of ['demoMode','localMode','pvpMode'])document.getElementById(id).innerHTML=`<option value="classic">${tr('mode_classic')}</option><option value="extended">${tr('mode_extended')}</option>`;document.getElementById('localMatch').innerHTML=`<option value="ranked">${tr('ranked')}</option><option value="casual">${tr('casual')}</option>`}
function renderStatus(){document.getElementById('status').textContent=state.username?`${tr('welcome')} ${state.username}`:tr('login_required');document.getElementById('who').textContent=state.username}
function setLanguage(v){state.lang=v;applyI18n();send({type:'set_language',language:v})}
function registerAccount(){send({type:'register',username:val('registerUser'),password:val('registerPass')})}
function login(){send({type:'login',username:val('loginUser'),password:val('loginPass')})}
function val(id){return document.getElementById(id).value.trim()}
function startAiDemo(){showView('game');send({type:'start_ai_demo',mode:val('demoMode'),rounds:Number(val('demoRounds')||1)})}
function startLocal(){showView('game');send({type:'start_local_ai_match',mode:val('localMode'),match_type:val('localMatch')})}
function createPvp(){send({type:'pvp_create_room',room_name:val('pvpRoomName'),password:val('pvpRoomPass'),mode:val('pvpMode'),max_rounds:Number(val('pvpRounds')||1)})}
function joinPvp(){send({type:'pvp_join_room',room_name:val('joinPvpName'),password:val('joinPvpPass')})}
function renderRooms(rooms){rooms=(rooms||[]).filter(Boolean);const box=document.getElementById('rooms');box.innerHTML=rooms.map((r,i)=>`<div class="room-card"><b>${r.room_name}</b> <span class="pill">${tr(r.mode_key)}</span> <span class="pill">${tr(r.status)||r.status}</span><div>${tr('round')}: ${r.current_round}/${r.max_rounds}</div><div>${tr('score')}: ${Object.entries(r.scores||{}).map(([n,s])=>n+':'+s).join(' ')}</div><div>${(r.seats||[]).map(s=>`${tr('seat')} ${s.seat+1}: ${s.username}`).join(' | ')}</div><button onclick="chooseRoom(${i})">${tr('join_room')}</button>${state.username===r.owner_username?` <button onclick="startRoomFromList(${i})">${tr('start_match')}</button> <button class="danger" onclick="disbandRoomFromList(${i})">${tr('disband_room')}</button>`:''}</div>`).join('');state.lastRooms=rooms}
function chooseRoom(i){const r=(state.lastRooms||[])[i];if(!r)return;document.getElementById('joinPvpName').value=r.room_name;joinPvp()}
function startRoomFromList(i){const r=(state.lastRooms||[])[i];if(!r)return;state.pvpRoom=r.room_name;send({type:'pvp_start_match',room_name:state.pvpRoom})}
function disbandRoomFromList(i){const r=(state.lastRooms||[])[i];if(!r)return;state.pvpRoom=r.room_name;send({type:'pvp_disband_room',room_name:r.room_name})}
function renderPlayers(players=[]){document.getElementById('players').innerHTML=players.map(p=>`<span class="pill">${tr('seat')} ${p.seat+1}: ${p.username||tr('empty')} ${tr(p.role||'farmer')} ${p.hand_size||0}</span>`).join('')}
function renderHand(){const h=document.getElementById('hand');h.innerHTML=state.cards.map((c,i)=>`<span id="card-${i}" class="card ${state.selected.has(i)?'selected':''}" onclick="toggleCard(${i})">${localizeCard(c)}</span>`).join('');document.getElementById('cardCount').textContent=state.cards.length}
function toggleCard(i){state.selected.has(i)?state.selected.delete(i):state.selected.add(i);renderHand()}
function actionButtons(html){document.getElementById('actionPanel').innerHTML=html}
function playSelected(){const cards=[...state.selected].sort((a,b)=>a-b);send({type:'play_card',action:'play',cards});state.selected.clear();renderHand()}
function handle(m){const p=m.payload||{};switch(m.type){
case'session_state':state.username=p.username||'';state.lang=p.language||state.lang;applyI18n();showView(state.username?'home':'auth');break;
case'auth_result':if(p.ok){state.username=p.username;showView('home')}logText(reason(p));applyI18n();break;
case'logout_result':state.username='';showView('auth');logLine('logged_out');applyI18n();break;
case'stats_result':showStats(p.stats);break;
case'rules_result':showRules(p);break;
case'pvp_rooms':renderRooms(p.rooms);break;
case'pvp_room':state.pvpRoom=p.room&&p.room.room_name;showView('pvp');if(p.room)renderRooms([p.room]);logText((p.ok?reason(p):`${tr('error')}: ${reason(p)}`));break;
case'pvp_room_disbanded':state.pvpRoom=null;state.roomId=null;state.seat=null;showView(state.username?'home':'auth');renderRooms([]);logText(reason(p));break;
case'pvp_match_started':state.pvpRoom=p.room&&p.room.room_name;showView('game');logText(p.ok?reason(p):`${tr('error')}: ${reason(p)}`);break;
case'local_game_created':state.roomId=p.room_id;state.seat=p.seat;showView('game');break;
case'room_created':case'room_joined':state.roomId=p.room_id;state.seat=p.seat;showView('game');logLine(m.type==='room_created'?'room_created':'room_joined');break;
case'room_state':renderPlayers(p.players);document.getElementById('gameInfo').textContent=`${tr(p.mode==='extended'?'mode_extended':'mode_classic')} · ${tr(p.state)}`;document.getElementById('startRoomBtn').classList.toggle('hidden',!(p.state==='waiting'&&p.host_username===state.username));break;
case'game_starting':showView('game');logText(`${tr('start_game')}: ${tr(p.mode==='extended'?'mode_extended':'mode_classic')}`);renderPlayers(p.players);break;
case'your_hand':case'your_cards':state.cards=p.cards||[];state.selected.clear();renderHand();break;
case'cards_dealt':logText(`${tr('marker_card')}: ${localizeCard({label:p.marked_card})} · ${tr('marker_holder')}: ${p.marker_holder_name||''} · ${tr('bottom_cards')}: ${p.bottom_count}`);break;
case'ask_bid':actionButtons((p.allowed_bids||[0,1,2,3]).map(b=>`<button onclick="send({type:'bid',bid:${b}});actionButtons('')">${b?tr('bid')+' '+b:tr('no_bid')}</button>`).join(''));break;
case'ask_call':actionButtons(`<button onclick="send({type:'call',call:true});actionButtons('')">${tr('call_landlord')}</button><button onclick="send({type:'call',call:false});actionButtons('')">${tr('dont_call')}</button>`);break;
case'ask_rob':actionButtons(`<button onclick="send({type:'rob',rob:true});actionButtons('')">${tr('rob_landlord')}</button><button onclick="send({type:'rob',rob:false});actionButtons('')">${tr('dont_rob')}</button>`);break;
case'ask_reveal':actionButtons(`<button onclick="send({type:'reveal',reveal:true});actionButtons('')">${tr('reveal')}</button><button onclick="send({type:'reveal',reveal:false});actionButtons('')">${tr('dont_reveal')}</button>`);break;
case'ask_report':actionButtons(`<button onclick="send({type:'report',report:true});actionButtons('')">${tr('report')}</button><button onclick="send({type:'report',report:false});actionButtons('')">${tr('dont_report')}</button>`);break;
case'ask_play':state.cards=p.hand||state.cards;renderHand();actionButtons(`<button onclick="playSelected()">${tr('play')}</button><button ${p.can_pass?'':'disabled'} onclick="send({type:'pass'});actionButtons('')">${tr('pass')}</button>`);break;
case'bid_result':logText(`${p.player_name}: ${p.bid?tr('bid')+' '+p.bid:tr('no_bid')}`);break;
case'call_result':logText(`${p.player_name}: ${yesNoText(p.call,'call_landlord','dont_call')}`);break;
case'rob_result':logText(`${p.player_name}: ${yesNoText(p.rob,'rob_landlord','dont_rob')}`);break;
case'reveal_result':logText(`${p.player_name}: ${yesNoText(p.reveal,'reveal','dont_reveal')}`);break;
case'report_result':logText(`${p.player_name}: ${p.report_label||tr('report')}`);break;
case'landlord_assigned':logText(`${p.player_name}: ${tr('landlord')} · ${cardsText(p.bottom_cards)}`);break;
case'play_turn':logText(`${tr('seat')} ${p.seat+1} ${p.player_name}: ${tr('play')}${p.is_opening?' · '+tr('round'):''}`);break;
case'new_round':logText(`${tr('round')}: ${p.leader_name}`);break;
case'no_bidder':logText(p.message||tr('no_bid'));break;
case'redeal':logText(p.message||tr('redeal_factor'));break;
case'player_empty':logText(`${p.player_name}: ${tr('winner')}`);break;
case'play_action':if(p.action==='play'){logText(`${p.player_name} ${tr('play')}: ${comboText(p.combo,p.combo_display)} -> ${cardsText(p.cards_played)} | ${tr('your_hand')}: ${p.remaining_count}`)}else{logText(`${p.player_name}: ${tr('pass')} | ${tr('your_hand')}: ${p.remaining_count}`)}break;
case'game_over':showSettlement(p);break;
case'pvp_round_result':if(p.room)renderRooms([p.room]);showSettlement(p);logLine(p.message_key||'pvp_round_finished');break;
case'stats_updated':logLine('stats_updated');break;
case'error':logText(`${tr('error')}: ${reason(p)}`);break;
}}
function showStats(s){if(!s)return;logLine('stats');document.getElementById('log').textContent+=`${tr('score')}: ${s.rating}\\n${tr('winner')}: ${s.wins}/${s.games}\\n`}
function showRules(p){showView('rules');document.getElementById('rulesContent').innerHTML=(p.sections||[]).map(s=>`<h3>${tr(s.title_key)}</h3><ul>${s.body_keys.map(k=>`<li>${tr(k)}</li>`).join('')}</ul>`).join('')}
function showSettlement(p){const s=p.settlement||{};logLine('finished');document.getElementById('log').textContent+=`${tr('winner')}: ${p.winner_name||''}\\n${tr('base_score')}: ${s.base_score||''} ${tr('multiplier')}: ${s.multiplier_factor||''} ${tr('total')}: ${s.total_score||''}\\n${tr('bomb')}: ${s.bomb_multiplier||0} ${tr('reveal_factor')}: ${s.reveal_multiplier||0} ${tr('redeal_factor')}: ${s.redeal_multiplier||0} ${tr('report_factor')}: ${s.report_multiplier||0} ${tr('marker_factor')}: ${s.marker_multiplier||0} ${tr('spring')}: ${s.spring_multiplier||0} ${tr('reverse_spring')}: ${s.reverse_spring_multiplier||0}\\n`}
function connect(){state.ws=new WebSocket((location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws');state.ws.onopen=()=>send({type:'get_session_state',language:state.lang});state.ws.onmessage=e=>handle(JSON.parse(e.data));state.ws.onclose=()=>setTimeout(connect,1200)}
document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key.toLowerCase()==='p'&&document.activeElement.classList.contains('password-input')){e.preventDefault();const el=document.activeElement;el.type=el.type==='password'?'text':'password'}});
applyI18n();connect();
</script>
</body>
</html>"""


# ============================================================
# HTTP Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    return INDEX_HTML


@app.get("/health")
async def health():
    return {"status": "ok", "rooms": len(manager.rooms)}


# ============================================================
# WebSocket Endpoint
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    current_room_id: str | None = None
    current_seat: int | None = None
    current_username: str = ""
    session = ClientSession(ws=ws)
    active_sessions[id(ws)] = session

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({
                    "type": "error",
                    "room_id": current_room_id or "",
                    "timestamp": "",
                    "request_id": None,
                    "payload": {"message": "Invalid JSON"},
                })
                continue

            msg_type = data.get("type", data.get("action", ""))
            request_id = data.get("request_id")
            if data.get("language") in SUPPORTED_LANGUAGES:
                session.language = data["language"]

            # ---- room management ----

            if msg_type == "get_session_state":
                await send_ws(ws, "session_state", {
                    "username": session.username,
                    "language": session.language,
                    "supported_languages": sorted(SUPPORTED_LANGUAGES),
                }, request_id)

            elif msg_type == "set_language":
                language = data.get("language", "zh")
                if language in SUPPORTED_LANGUAGES:
                    session.language = language
                await send_ws(ws, "session_state", {
                    "username": session.username,
                    "language": session.language,
                    "supported_languages": sorted(SUPPORTED_LANGUAGES),
                }, request_id)

            elif msg_type == "register":
                username = (data.get("username") or "").strip()
                password = data.get("password") or ""
                try:
                    ok, message = accounts.register(username, password)
                except SupabaseError as exc:
                    await send_ws(ws, "auth_result", {"ok": False, "message_key": "error", "message": str(exc)}, request_id)
                    continue
                if ok:
                    session.username = username
                    current_username = username
                    active_users[username] = session
                await send_ws(ws, "auth_result", {
                    "ok": ok,
                    "username": username if ok else "",
                    "message": message,
                    "message_key": "register_success" if ok else "error",
                    "stats": public_stats(username) if ok else None,
                }, request_id)

            elif msg_type == "login":
                username = (data.get("username") or "").strip()
                password = data.get("password") or ""
                try:
                    ok, message = accounts.authenticate(username, password)
                except SupabaseError as exc:
                    await send_ws(ws, "auth_result", {"ok": False, "message_key": "error", "message": str(exc)}, request_id)
                    continue
                if ok:
                    session.username = username
                    current_username = username
                    active_users[username] = session
                await send_ws(ws, "auth_result", {
                    "ok": ok,
                    "username": username if ok else "",
                    "message": message,
                    "message_key": "login_success" if ok else "error",
                    "stats": public_stats(username) if ok else None,
                }, request_id)

            elif msg_type == "logout":
                if session.username and active_users.get(session.username) is session:
                    active_users.pop(session.username, None)
                session.username = ""
                current_username = ""
                await send_ws(ws, "logout_result", {"message_key": "logged_out"}, request_id)

            elif msg_type == "get_stats":
                if not session.username:
                    await send_ws(ws, "error", {"message_key": "login_required"}, request_id)
                    continue
                await send_ws(ws, "stats_result", {"stats": public_stats(session.username)}, request_id)

            elif msg_type == "get_rules":
                await send_ws(ws, "rules_result", rules_payload(), request_id)

            elif msg_type == "start_ai_demo":
                mode = data.get("mode", "classic")
                if mode not in MODE_RULES:
                    mode = "classic"
                room = manager.create_room(mode, "AI-Demo")
                room.base_score = LOCAL_RANKED_BASE_SCORE
                room.match_kind = "casual_no_score"
                room.add_observer(ws)
                room.fill_with_ai()
                await send_ws(ws, "local_game_created", {"room_id": room.room_id, "seat": None, "mode": mode, "match_type": "demo"}, request_id)
                await room.start_game()

            elif msg_type == "start_local_ai_match":
                await start_local_ai_game(
                    session,
                    data.get("mode", "classic"),
                    data.get("match_type", "casual"),
                    request_id,
                )

            elif msg_type == "pvp_create_room":
                if not session.username:
                    await send_ws(ws, "error", {"message_key": "login_required"}, request_id)
                    continue
                max_rounds_raw = data.get("max_rounds", 1)
                try:
                    max_rounds = int(max_rounds_raw or 1)
                except (TypeError, ValueError):
                    max_rounds = 1
                try:
                    ok, message, room = pvp_manager.create_room(
                        session.username,
                        data.get("room_name", ""),
                        data.get("password", ""),
                        data.get("mode", "classic"),
                        max_rounds,
                    )
                except SupabaseError as exc:
                    await send_ws(ws, "error", {"message_key": "error", "message": str(exc)}, request_id)
                    continue
                except Exception as exc:
                    await send_ws(ws, "error", {"message_key": "error", "message": f"创建房间失败: {exc}"}, request_id)
                    continue
                if ok and room:
                    session.pvp_room_name = room["room_name"]
                await send_ws(ws, "pvp_room", {
                    "ok": ok,
                    "message": message,
                    "message_key": "room_created" if ok else "error",
                    "room": pvp_public_room(room) if room else None,
                }, request_id)

            elif msg_type == "pvp_list_rooms":
                try:
                    rooms = [pvp_public_room(room) for room in pvp_manager.list_rooms()]
                except SupabaseError as exc:
                    await send_ws(ws, "error", {"message_key": "error", "message": str(exc)}, request_id)
                    continue
                await send_ws(ws, "pvp_rooms", {"rooms": rooms}, request_id)

            elif msg_type == "pvp_join_room":
                if not session.username:
                    await send_ws(ws, "error", {"message_key": "login_required"}, request_id)
                    continue
                try:
                    ok, message, room = pvp_manager.join_room(
                        session.username,
                        data.get("room_name", ""),
                        data.get("password", ""),
                    )
                except SupabaseError as exc:
                    await send_ws(ws, "error", {"message_key": "error", "message": str(exc)}, request_id)
                    continue
                except Exception as exc:
                    await send_ws(ws, "error", {"message_key": "error", "message": f"加入房间失败: {exc}"}, request_id)
                    continue
                if ok and room:
                    session.pvp_room_name = room["room_name"]
                    live_id = pvp_live_rooms.get(room["room_name"])
                    live = manager.get_room(live_id) if live_id else None
                    if live and live.state == "playing":
                        for idx, seat_info in live.seats.items():
                            if seat_info.username == session.username:
                                await attach_session_to_room(session, live, idx)
                                break
                await send_ws(ws, "pvp_room", {
                    "ok": ok,
                    "message": message,
                    "message_key": "room_joined" if ok else "error",
                    "room": pvp_public_room(room) if room else None,
                }, request_id)

            elif msg_type == "pvp_start_match":
                if not session.username:
                    await send_ws(ws, "error", {"message_key": "login_required"}, request_id)
                    continue
                room_name = data.get("room_name") or session.pvp_room_name or ""
                try:
                    ok, message, room = pvp_manager.start_room(session.username, room_name)
                    if ok and room:
                        started, start_key, public_room = await start_bound_pvp_round(room_name, room)
                        ok = started
                        message = start_key
                    else:
                        public_room = pvp_public_room(room) if room else None
                except SupabaseError as exc:
                    await send_ws(ws, "error", {"message_key": "error", "message": str(exc)}, request_id)
                    continue
                except Exception as exc:
                    await send_ws(ws, "error", {"message_key": "error", "message": f"开始比赛失败: {exc}"}, request_id)
                    continue
                await send_ws(ws, "pvp_match_started" if ok else "error", {
                    "ok": ok,
                    "message_key": message if ok else "error",
                    "room": public_room,
                }, request_id)

            elif msg_type == "pvp_disband_room":
                if not session.username:
                    await send_ws(ws, "error", {"message_key": "login_required"}, request_id)
                    continue
                try:
                    ok, message = await disband_pvp_room_for_all(
                        data.get("room_name") or session.pvp_room_name or "",
                        session.username,
                    )
                except SupabaseError as exc:
                    await send_ws(ws, "error", {"message_key": "error", "message": str(exc)}, request_id)
                    continue
                await send_ws(ws, "pvp_room_disbanded" if ok else "error", {
                    "ok": ok,
                    "message": message,
                    "message_key": "room_disbanded" if ok else "error",
                    "room": None,
                }, request_id)

            elif msg_type == "create_room":
                username = (data.get("username") or "player").strip()[:30]
                mode = data.get("mode", "classic")
                if mode not in MODE_RULES:
                    mode = "classic"
                room = manager.create_room(mode, username)
                seat = room.add_player(username)
                if seat is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": "",
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "Failed to join room"},
                    })
                    continue
                current_room_id = room.room_id
                current_seat = seat
                current_username = username
                session.room_id = room.room_id
                session.seat = seat
                room.seats[seat].ws = ws
                room._start_sender(seat)
                manager.register_ws(ws, room.room_id, seat, username)

                await room._send_to(seat, "room_created", {
                    "room_id": room.room_id,
                    "seat": seat,
                }, request_id=request_id)
                await room._broadcast("room_state", room.public_room_state()["payload"])

            elif msg_type == "join_room":
                room_id = data.get("room_id", "").strip().upper()
                username = (data.get("username") or "player").strip()[:30]
                room = manager.get_room(room_id)
                if room is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "房间不存在"},
                    })
                    continue
                if room.state != "waiting":
                    await ws.send_json({
                        "type": "error",
                        "room_id": room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "游戏已开始"},
                    })
                    continue
                seat = room.add_player(username)
                if seat is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "房间已满"},
                    })
                    continue
                current_room_id = room_id
                current_seat = seat
                current_username = username
                session.room_id = room_id
                session.seat = seat
                room.seats[seat].ws = ws
                room._start_sender(seat)
                manager.register_ws(ws, room_id, seat, username)

                await room._send_to(seat, "room_joined", {
                    "room_id": room_id,
                    "seat": seat,
                }, request_id=request_id)
                await room._broadcast("room_state", room.public_room_state()["payload"])

            elif msg_type == "start_game":
                room = manager.get_room(current_room_id) if current_room_id else None
                if room is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": current_room_id or "",
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "未在房间中"},
                    })
                    continue
                if room.host_username != (room.seats.get(current_seat, SeatInfo(username="")).username if current_seat is not None else ""):
                    await ws.send_json({
                        "type": "error",
                        "room_id": current_room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "只有房主可以开始"},
                    })
                    continue
                if not room.all_seats_filled():
                    room.fill_with_ai()
                if not any(s.is_human and s.connected for s in room.seats.values()):
                    await ws.send_json({
                        "type": "error",
                        "room_id": current_room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "至少需要一名真人玩家"},
                    })
                    continue
                await room.start_game()

            elif msg_type == "leave_room":
                if current_room_id:
                    room = manager.get_room(current_room_id)
                    if room and current_seat is not None:
                        await room._stop_sender(current_seat)
                        room.remove_player(room.seats[current_seat].username)
                        manager.unregister_ws(ws)
                        await room._broadcast("room_state", room.public_room_state()["payload"])
                        human_count = sum(1 for s in room.seats.values() if not s.is_human or s.connected)
                        if human_count == 0:
                            manager.remove_room(current_room_id)
                current_room_id = None
                current_seat = None
                current_username = ""

            # ---- game actions ----

            elif msg_type in ("bid", "play_card", "pass", "call", "rob", "reveal", "report"):
                active_room_id = session.room_id or current_room_id
                active_seat = session.seat if session.seat is not None else current_seat
                room = manager.get_room(active_room_id) if active_room_id else None
                if room is None or active_seat is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": active_room_id or "",
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "未在游戏中"},
                    })
                    continue

                if msg_type == "bid":
                    room.handle_response(active_seat, {
                        "bid": data.get("bid", data.get("score", 0)),
                    })
                elif msg_type == "call":
                    room.handle_response(active_seat, {
                        "call": data.get("call", False),
                    })
                elif msg_type == "rob":
                    room.handle_response(active_seat, {
                        "rob": data.get("rob", False),
                    })
                elif msg_type == "reveal":
                    room.handle_response(active_seat, {
                        "reveal": data.get("reveal", False),
                    })
                elif msg_type == "report":
                    room.handle_response(active_seat, {
                        "report": data.get("report", False),
                    })
                elif msg_type == "pass":
                    room.handle_response(active_seat, {
                        "action": "pass",
                        "cards": [],
                    })
                elif msg_type == "play_card":
                    room.handle_response(active_seat, {
                        "action": data.get("action", "play"),
                        "cards": data.get("cards", []),
                    })

            # ---- recovery / reconnection ----

            elif msg_type == "reconnect":
                token = data.get("recovery_token", "")
                result = await manager.reconnect_player(token, ws)
                if result is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": "",
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "重连失败，token无效或已过期。"},
                    })
                    continue
                current_room_id = result["room_id"]
                session.room_id = current_room_id
                # Reconstruct seat from snapshot
                for p in result["payload"].get("players", []):
                    if p.get("username") == data.get("username"):
                        current_seat = p["seat"]
                        current_username = p["username"]
                        session.seat = current_seat
                        break
                await ws.send_json(result)

            elif msg_type == "request_snapshot":
                active_room_id = session.room_id or current_room_id
                active_seat = session.seat if session.seat is not None else current_seat
                room = manager.get_room(active_room_id) if active_room_id else None
                if room is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": active_room_id or "",
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "未在房间中"},
                    })
                    continue
                snapshot = room.full_state_snapshot(for_seat=active_seat)
                await ws.send_json(snapshot)

            else:
                await ws.send_json({
                    "type": "error",
                    "room_id": current_room_id or "",
                    "timestamp": "",
                    "request_id": request_id,
                    "payload": {"message": f"未知操作: {msg_type}"},
                })

    except WebSocketDisconnect:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        active_sessions.pop(id(ws), None)
        if session.username and active_users.get(session.username) is session:
            active_users.pop(session.username, None)
        # Cleanup on disconnect via ConnectionManager (generates recovery token)
        cleanup_room_id = session.room_id or current_room_id
        cleanup_seat = session.seat if session.seat is not None else current_seat
        if cleanup_room_id is not None and cleanup_seat is not None:
            try:
                await manager.handle_disconnect(ws)
            except Exception:
                pass


# ============================================================
# Command Line Game Shell
# ============================================================

def prompt_choice(title: str, choices: list[tuple[str, str]]) -> str:
    print(f"\n{title}")
    for key, label in choices:
        print(f"{key}. {label}")
    allowed = {key for key, _ in choices}
    while True:
        choice = input("请选择: ").strip()
        if choice in allowed:
            return choice
        print("选项无效，请重新输入。")


def prompt_int(prompt: str, default: int | None = None, minimum: int | None = None, maximum: int | None = None) -> int:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(prompt + suffix + ": ").strip()
        if not raw and default is not None:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("请输入数字。")
                continue
        if minimum is not None and value < minimum:
            print(f"不能小于 {minimum}。")
            continue
        if maximum is not None and value > maximum:
            print(f"不能大于 {maximum}。")
            continue
        return value


def choose_mode() -> str:
    choice = prompt_choice("选择玩法", [("1", "二打一"), ("2", "三打一")])
    return "classic" if choice == "1" else "extended"


def build_players(mode: str, username: str | None = None) -> list[Player]:
    player_count = MODE_RULES[mode]["player_count"]
    if username is None:
        return [Player(f"AI-{index + 1}", is_human=False) for index in range(player_count)]
    players = [Player(username, is_human=True, account_username=username)]
    players.extend(Player(f"AI-{index}", is_human=False) for index in range(1, player_count))
    return players


def print_settlement(result: dict, players: list[Player], match_type: str) -> None:
    settlement = result["settlement"]
    print("\n结算明细")
    print(f"胜方: {result['winner_role']} | 地主: {result['landlord_name']} | 赢家: {result['winner_name']}")
    calculated_total = settlement["base_score"] * settlement["bid_value"] * settlement["multiplier_factor"]
    print(
        f"基本分 {settlement['base_score']} x 叫分 {settlement['bid_value']} "
        f"x 倍率 {settlement['multiplier_factor']} = {calculated_total}"
    )
    print(
        "加倍项: 炸弹/王炸 {bomb_multiplier}, 摊打 {reveal_multiplier}, 荒番 {redeal_multiplier}, "
        "报道 {report_multiplier}, 标记王 {marker_multiplier}, 春天 {spring_multiplier}, "
        "反春天 {reverse_spring_multiplier}".format(**settlement)
    )
    if match_type == "casual":
        print("娱乐赛只记录胜负，不改变积分。")
        return
    total = int(settlement["total_score"])
    farmer_count = len(players) - 1
    for player in players:
        won = player.role == result["winner_role"]
        if player.role == "landlord":
            delta = farmer_count * total if won else -farmer_count * total
        else:
            delta = total if won else -total
        print(f"{player.name}: {'+' if delta >= 0 else ''}{delta}")


def rating_delta_for_player(result: dict, players: list[Player], username: str) -> int:
    settlement = result["settlement"]
    total = int(settlement["total_score"])
    if total == 0:
        return 0
    farmer_count = len(players) - 1
    player = next(player for player in players if player.account_username == username)
    won = player.role == result["winner_role"]
    if player.role == "landlord":
        return farmer_count * total if won else -farmer_count * total
    return total if won else -total


def run_ai_demo(mode: str | None = None, rounds: int | None = None) -> None:
    mode = mode or choose_mode()
    rounds = rounds if rounds is not None else prompt_int("要演示几轮", default=1, minimum=1, maximum=50)
    if rounds < 1:
        print("演示轮数至少为 1，已按 1 轮运行。")
        rounds = 1
    for round_no in range(1, rounds + 1):
        print(f"\n========== AI 演示第 {round_no}/{rounds} 轮：{MODE_RULES[mode]['label']} ==========")
        players = build_players(mode)
        result = GameSession(mode, players, match_type="casual", god_view=True).run()
        print_settlement(result, players, "casual")


def run_local_ai_match(accounts: AccountManager, username: str) -> None:
    mode = choose_mode()
    match_choice = prompt_choice("选择本地 AI 对战类型", [("1", "积分赛"), ("2", "娱乐赛")])
    match_type = "ranked" if match_choice == "1" else "casual"
    if match_type == "ranked":
        ok, message, stats = accounts.prepare_ranked_entry(username)
        print(message)
        if stats:
            print(f"当前积分: {stats['rating']}")
        if not ok:
            return

    players = build_players(mode, username)
    result = GameSession(mode, players, match_type=match_type).run()
    print_settlement(result, players, match_type)
    human = players[0]
    won = human.role == result["winner_role"]
    delta = rating_delta_for_player(result, players, username) if match_type == "ranked" else 0
    accounts.record_result(username, won, match_type=match_type, rating_delta=delta)
    stats = accounts.get_user_stats(username)
    if stats:
        print(f"\n已更新战绩。当前积分 {stats['rating']}，总战绩 {stats['wins']}/{stats['games']}。")


def show_user_stats(accounts: AccountManager, username: str) -> None:
    stats = accounts.get_user_stats(username)
    if not stats:
        print("没有找到当前账号战绩。")
        return
    print("\n我的战绩")
    print(f"账号: {stats['username']} | 排名: {stats['rank']} | 当前积分: {stats['rating']}")
    print(f"总战绩: {stats['wins']}胜 {stats['losses']}负，胜率 {stats['win_rate']:.1%}")
    print(f"积分赛: {stats['ranked_wins']}胜 {stats['ranked_losses']}负，胜率 {stats['ranked_win_rate']:.1%}")
    print(f"娱乐赛: {stats['casual_wins']}胜 {stats['casual_losses']}负，胜率 {stats['casual_win_rate']:.1%}")
    print(f"今日补分: {stats['daily_replenish_used']}/{DAILY_REPLENISH_LIMIT}")


def print_pvp_room(room: dict) -> None:
    if not room:
        return
    seats = ", ".join(f"{seat['seat'] + 1}:{seat['username']}" for seat in sorted(room.get("seats") or [], key=lambda item: item["seat"]))
    scores = ", ".join(f"{name}:{score}" for name, score in sorted((room.get("scores") or {}).items()))
    print(
        f"{room['room_name']} | 房主 {room['owner_username']} | {MODE_RULES[room['mode']]['label']} | "
        f"{room['status']} | 第 {room['current_round']}/{room['max_rounds']} 轮"
    )
    print(f"玩家: {seats or '无'}")
    print(f"分数: {scores or '暂无'}")
    if room.get("winner_username"):
        print(f"比赛胜利者: {room['winner_username']}")


def list_pvp_rooms(pvp: PvpManager) -> None:
    rooms = pvp.list_rooms()
    if not rooms:
        print("当前没有可显示的 PVP 房间。")
        return
    print("\n线上 PVP 房间")
    for room in rooms:
        print_pvp_room(room)
        print("-" * 40)


def pvp_score_unit() -> int:
    bid = prompt_int("叫分/基础倍数", default=1, minimum=1, maximum=3)
    names = ["炸弹/王炸", "摊打", "荒番", "报道", "春天", "反春天", "标记王"]
    double_count = 0
    for name in names:
        double_count += prompt_int(f"{name}加倍次数", default=0, minimum=0, maximum=20)
    score = bid * (2**double_count)
    print(f"线上 PVP 基本分 1 x 叫分 {bid} x 2^{double_count} = 本轮每农民单位 {score} 分。")
    return score


def run_pvp_menu(username: str) -> None:
    pvp = PvpManager()
    while True:
        choice = prompt_choice(
            "线上 PVP",
            [
                ("1", "创建房间"),
                ("2", "搜索房间"),
                ("3", "加入房间"),
                ("4", "开始比赛"),
                ("5", "房主录入结算"),
                ("6", "返回"),
            ],
        )
        try:
            if choice == "1":
                room_name = input("房间名: ").strip()
                password = input("房间密码（可空）: ").strip()
                mode = choose_mode()
                max_rounds = prompt_int("比赛轮数", default=1, minimum=1, maximum=50)
                ok, message, room = pvp.create_room(username, room_name, password, mode, max_rounds)
                print(message)
                if ok and room:
                    print_pvp_room(room)
            elif choice == "2":
                list_pvp_rooms(pvp)
            elif choice == "3":
                room_name = input("要加入的房间名: ").strip()
                password = input("房间密码（无密码可空）: ").strip()
                ok, message, room = pvp.join_room(username, room_name, password)
                print(message)
                if room:
                    print_pvp_room(room)
            elif choice == "4":
                room_name = input("要开始的房间名: ").strip()
                ok, message, room = pvp.start_room(username, room_name)
                print(message)
                if room:
                    print_pvp_room(room)
            elif choice == "5":
                room_name = input("房间名: ").strip()
                landlord = input("本轮地主用户名: ").strip()
                won_choice = prompt_choice("地主是否获胜", [("1", "地主胜"), ("2", "农民胜")])
                score_unit = pvp_score_unit()
                ok, message, room = pvp.record_final_result(
                    room_name,
                    username,
                    landlord,
                    landlord_won=won_choice == "1",
                    multiplier=score_unit,
                )
                print(message)
                if room:
                    print_pvp_room(room)
            else:
                return
        except SupabaseError as exc:
            print(f"线上服务暂时不可用: {exc}")


def authenticated_menu(accounts: AccountManager, username: str) -> None:
    while True:
        choice = prompt_choice(
            f"{username}，请选择",
            [("1", "本地 AI 对战"), ("2", "查看战绩"), ("3", "线上 PVP"), ("4", "退出登录")],
        )
        try:
            if choice == "1":
                run_local_ai_match(accounts, username)
            elif choice == "2":
                show_user_stats(accounts, username)
            elif choice == "3":
                run_pvp_menu(username)
            else:
                return
        except SupabaseError as exc:
            print(f"账号服务暂时不可用: {exc}")


def register_account(accounts: AccountManager) -> str | None:
    username = input("用户名（3-20位）: ").strip()
    password = input("密码（至少6位）: ").strip()
    ok, message = accounts.register(username, password)
    print(message)
    return username if ok else None


def login_account(accounts: AccountManager) -> str | None:
    username = input("用户名: ").strip()
    password = input("密码: ").strip()
    ok, message = accounts.authenticate(username, password)
    print(message)
    return username if ok else None


def run_cli(args: argparse.Namespace) -> None:
    if args.demo:
        run_ai_demo(args.demo, args.rounds)
        return

    accounts = AccountManager()
    print("欢迎来到斗地主。")
    print(
        f"积分赛规则: 初始 {STARTING_RATING} 分，至少 {RANKED_MIN_RATING} 分可参赛；"
        f"低于门槛时每天最多自动补分 {DAILY_REPLENISH_LIMIT} 次。"
    )
    while True:
        choice = prompt_choice("账号入口", [("1", "注册账号"), ("2", "登录账号"), ("3", "AI 对战"), ("4", "退出")])
        try:
            if choice == "1":
                username = register_account(accounts)
                if username:
                    authenticated_menu(accounts, username)
            elif choice == "2":
                username = login_account(accounts)
                if username:
                    authenticated_menu(accounts, username)
            elif choice == "3":
                run_ai_demo()
            else:
                print("再见。")
                return
        except SupabaseError as exc:
            print(f"账号服务暂时不可用: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="斗地主")
    parser.add_argument("--server", action="store_true", help="兼容旧参数；默认已经启动 WebSocket Web UI")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="监听地址")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")), help="监听端口")
    parser.add_argument("--no-open", action="store_true", help="启动后不自动打开浏览器")
    parser.add_argument("--demo", choices=sorted(MODE_RULES), help="兼容旧参数；请在 Web UI 中启动 AI 演示")
    parser.add_argument("--rounds", type=int, help="兼容旧参数")
    return parser.parse_args()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    import uvicorn

    if not args.no_open:
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    uvicorn.run("main:app", host=args.host, port=args.port, log_level="info")
