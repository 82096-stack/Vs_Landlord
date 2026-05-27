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
        landlord_won = players[winner_idx].role == "landlord"
        ok, _message, updated = pvp_manager.record_final_result(
            room_name,
            supabase_room["owner_username"],
            landlord,
            landlord_won,
            max(1, int(settlement["total_score"])),
        )
        # Record PVP stats for each human player
        for player in players:
            if player.is_human:
                player_won = (player.role == "landlord") == landlord_won
                accounts.record_pvp_result(player.name, player_won)
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
 es:{app_title:'Dou Dizhu WebSocket',login:'Iniciar sesión',register:'Registrarse',username:'Usuario',password:'Contraseña',password_hint:'Presiona Control+P en este campo para mostrar u ocultar la contraseña.',ai_demo:'Partida IA',start_demo:'Iniciar demo',welcome:'Bienvenido',local_ai:'Partida local con IA',stats:'Estadísticas',online_pvp:'PVP en línea',rules:'Reglas',logout:'Cerrar sesión',start_game:'Iniciar juego',back:'Volver',create_room:'Crear sala',join_room:'Unirse a sala',disband_room:'Disolver sala',room_name:'Nombre de sala',room_password_optional:'Contraseña (opcional)',refresh:'Actualizar',rooms:'Salas',game_table:'Mesa de juego',your_hand:'Tu mano',log:'Registro',mode_classic:'Dos contra uno',mode_extended:'Tres contra uno',ranked:'Competitivo',casual:'Casual',start_match:'Iniciar partida',seat:'Asiento',empty:'Vacío',landlord:'Terrateniente',farmer:'Campesino',connected:'Conectado',disconnected:'Desconectado',waiting:'Esperando',playing:'Jugando',finished:'Terminado',completed:'Completado',lobby:'Vestíbulo',bid:'Pujar',no_bid:'No pujar',call_landlord:'Ser terrateniente',dont_call:'No ser',rob_landlord:'Robar terrateniente',dont_rob:'No robar',reveal:'Mostrar',dont_reveal:'No mostrar',report:'Informar',dont_report:'No informar',play:'Jugar',pass:'Pasar',round:'Ronda',score:'Puntuación',winner:'Ganador',base_score:'Puntuación base',multiplier:'Multiplicador',bomb:'Bomba/Cohete',reveal_factor:'Mostrar',redeal_factor:'Redistribuir',report_factor:'Informar',marker_factor:'Comodín marcador',marker_card:'Carta marcada',marker_holder:'Portador de carta marcada',bottom_cards:'Cartas inferiores',spring:'Primavera',reverse_spring:'Primavera inversa',total:'Total',small_joker:'Comodín pequeño',big_joker:'Comodín grande',single:'Individual',pair:'Par',trio:'Trío',trio_single:'Trío con individual',trio_pair:'Trío con par',straight:'Escalera',pair_straight:'Escalera de pares',trio_straight:'Avión',airplane_single:'Avión con individuales',airplane_pair:'Avión con pares',four_two_single:'Cuatro con dos',four_two_pair:'Cuatro con dos pares',rocket:'Cohete',room_created:'Sala creada.',room_joined:'Te has unido a la sala.',room_disbanded:'Sala disuelta. Todos los jugadores han vuelto al vestíbulo.',login_success:'Inicio de sesión exitoso.',register_success:'Registro exitoso.',logged_out:'Sesión cerrada.',login_required:'Por favor inicia sesión primero.',stats_updated:'Estadísticas actualizadas.',pvp_started:'Partida PVP iniciada.',pvp_round_finished:'Ronda PVP terminada.',pvp_match_finished:'Partida PVP terminada.',ranked_entry_ok:'Entrada competitiva aceptada.',ranked_entry_denied:'Entrada competitiva denegada.',room_join_failed:'No se pudo unir a la sala.',pvp_already_playing:'Esta partida PVP ya está en juego.',pvp_seat_failed:'No se pudieron crear los asientos PVP.',error:'Error',rules_classic_title:'Objetivo: dos campesinos contra un terrateniente',rules_classic_1:'El modo clásico tiene 3 jugadores. Uno es el terrateniente y los otros dos son campesinos del mismo equipo.',rules_classic_2:'La baraja tiene 54 cartas. Cada jugador recibe 17 cartas y 3 cartas quedan boca abajo para el futuro terrateniente.',rules_classic_3:'El jugador que recibe la carta marcada inicia la decisión. Un jugador puede pedir ser terrateniente, luego otros pueden robarlo.',rules_classic_4:'Después de la puja, el terrateniente toma las 3 cartas inferiores y juega primero.',rules_classic_5:'El terrateniente gana vaciando su mano primero. Los campesinos ganan si cualquiera de ellos vacía su mano primero.',rules_extended_title:'Modo tres contra uno',rules_extended_1:'El modo extendido tiene 4 jugadores. Un terrateniente lucha contra tres campesinos.',rules_extended_2:'Usa dos barajas. Cada jugador recibe 25 cartas y el terrateniente recibe 8 cartas inferiores.',rules_extended_3:'El terrateniente puede mostrar su mano. Mostrar aumenta el riesgo pero también el multiplicador.',rules_extended_4:'Algunas manos pueden informar. Informar y doble informar añaden multiplicadores extra.',rules_extended_5:'Los campesinos tienen límites de bombas según la puja.',rules_extended_6:'Las escaleras son más flexibles en este modo, incluyendo secuencias A2345.',rules_turn_title:'Cómo funciona un turno',rules_turn_1:'El primer jugador de una baza puede jugar cualquier combinación legal de su mano.',rules_turn_2:'El siguiente jugador debe jugar el mismo tipo de combinación con un rango mayor, o pasar.',rules_turn_3:'Las bombas vencen a las combinaciones normales. Los cohetes vencen a las bombas y a las combinaciones normales.',rules_turn_4:'Si todos los demás pasan, el último jugador que jugó cartas inicia una nueva baza.',rules_turn_5:'Selecciona cartas de tu mano y presiona Jugar. Si no puedes o no quieres superar la mesa, presiona Pasar.',rules_combo_title:'Combinaciones comunes',rules_combo_1:'Individual: una carta. Par: dos cartas del mismo rango. Trío: tres cartas del mismo rango.',rules_combo_2:'Trío con individual o par: tres iguales más una carta extra o un par.',rules_combo_3:'Escalera: al menos cinco cartas individuales consecutivas. Escalera de pares: al menos tres pares consecutivos.',rules_combo_4:'Avión: tríos consecutivos, opcionalmente con individuales o pares extra.',rules_combo_5:'Bomba: cuatro o más cartas del mismo rango. Cohete: comodines juntos, la combinación más fuerte.',rules_scoring_title:'Puntuación',rules_scoring_1:'Las partidas competitivas locales comienzan con 1200 puntos y requieren al menos 200 puntos para entrar.',rules_scoring_2:'Si tu puntuación está por debajo de 200, el juego puede rellenarte a 1200 puntos hasta dos veces al día.',rules_scoring_3:'La puntuación base competitiva es 50. La puntuación base PVP es 1.',rules_scoring_4:'La puja es el multiplicador base. Bombas, cohetes, mostrar, redistribuciones, informes, primavera, primavera inversa y comodín marcador pueden duplicar la puntuación.',rules_scoring_5:'Las partidas casuales registran victorias y derrotas pero no cambian la puntuación.',rules_pvp_title:'PVP en línea',rules_pvp_1:'El propietario crea una sala, opcionalmente establece una contraseña, elige dos-contra-uno o tres-contra-uno, y elige el número de rondas.',rules_pvp_2:'Otros jugadores buscan en la lista de salas, ingresan la contraseña si es necesario, y se unen a la sala.',rules_pvp_3:'Cuando los asientos están llenos, el propietario inicia la partida. Los jugadores juegan por turnos a través de la interfaz web.',rules_pvp_4:'Después de cada ronda, las puntuaciones se calculan automáticamente. Cuando todas las rondas terminan, la puntuación total más alta gana.',rank:'Clasificación',room_name_length:'El nombre de la sala debe tener entre 2 y 30 caracteres.',invalid_mode:'Modo de juego no válido.',invalid_rounds:'Las rondas deben estar entre 1 y 50.',room_name_exists:'El nombre de la sala ya existe. Por favor elige otro.',room_not_found:'Sala no encontrada.',room_not_lobby:'La sala ya ha comenzado o terminado, no se puede unir.',wrong_password:'Contraseña de sala incorrecta.',already_in_room:'Ya estás en esta sala.',room_full:'Sala llena.',not_owner:'Solo el propietario de la sala puede realizar esta acción.',room_already_started:'La sala no se puede iniciar de nuevo.',not_enough_players:'No hay suficientes jugadores.',room_not_playing:'La sala no está actualmente en juego.',landlord_not_in_room:'El nombre de usuario del terrateniente no está en la sala.',match_completed:'Partida completada.',round_finished:'Ronda terminada.',player_disconnected:'{username} se desconectó.',player_reconnected:'{username} se reconectó.',all_disconnected:'Todos los jugadores se desconectaron. La sala se limpiará en 5 minutos.',invalid_combo:'Combinación de cartas no válida.',not_your_turn:'No es tu turno.',seat_taken:'El asiento ya está ocupado.',room_full_players:'La sala está llena.',game_already_started:'El juego ya ha comenzado.',cannot_pass_first:'Eres el primer jugador de esta baza y no puedes pasar.',cards_not_in_hand:'Algunas cartas seleccionadas no están en tu mano.',must_play_higher:'Debes jugar una combinación más alta del mismo tipo.',landlord_not_selected:'El terrateniente aún no ha sido seleccionado.',bidding_in_progress:'La puja está en progreso.',not_in_room:'No estás en esta sala.'},
 fr:{app_title:'Dou Dizhu WebSocket',login:'Connexion',register:'Créer un compte',username:'Nom d\\'utilisateur',password:'Mot de passe',password_hint:'Appuyez sur Control+P dans ce champ pour afficher ou masquer le mot de passe.',ai_demo:'Partie IA',start_demo:'Démarrer la démo',welcome:'Bienvenue',local_ai:'Partie locale avec IA',stats:'Statistiques',online_pvp:'PVP en ligne',rules:'Règles',logout:'Déconnexion',start_game:'Démarrer le jeu',back:'Retour',create_room:'Créer une salle',join_room:'Rejoindre une salle',disband_room:'Dissoudre la salle',room_name:'Nom de la salle',room_password_optional:'Mot de passe (optionnel)',refresh:'Actualiser',rooms:'Salles',game_table:'Table de jeu',your_hand:'Votre main',log:'Journal',mode_classic:'Deux contre un',mode_extended:'Trois contre un',ranked:'Classé',casual:'Amical',start_match:'Démarrer la partie',seat:'Siège',empty:'Vide',landlord:'Propriétaire',farmer:'Paysan',connected:'Connecté',disconnected:'Déconnecté',waiting:'En attente',playing:'En cours',finished:'Terminé',completed:'Achevé',lobby:'Hall',bid:'Enchère',no_bid:'Pas d\\'enchère',call_landlord:'Devenir propriétaire',dont_call:'Ne pas devenir',rob_landlord:'Voler le propriétaire',dont_rob:'Ne pas voler',reveal:'Révéler',dont_reveal:'Ne pas révéler',report:'Rapporter',dont_report:'Ne pas rapporter',play:'Jouer',pass:'Passer',round:'Tour',score:'Score',winner:'Gagnant',base_score:'Score de base',multiplier:'Multiplicateur',bomb:'Bombe/Fusée',reveal_factor:'Révéler',redeal_factor:'Redistribution',report_factor:'Rapport',marker_factor:'Marqueur joker',marker_card:'Carte marquée',marker_holder:'Porteur de carte marquée',bottom_cards:'Cartes du bas',spring:'Printemps',reverse_spring:'Printemps inversé',total:'Total',small_joker:'Petit joker',big_joker:'Grand joker',single:'Simple',pair:'Paire',trio:'Trio',trio_single:'Trio avec simple',trio_pair:'Trio avec paire',straight:'Suite',pair_straight:'Suite de paires',trio_straight:'Avion',airplane_single:'Avion avec simples',airplane_pair:'Avion avec paires',four_two_single:'Quatre avec deux',four_two_pair:'Quatre avec deux paires',rocket:'Fusée',room_created:'Salle créée.',room_joined:'Vous avez rejoint la salle.',room_disbanded:'Salle dissoute. Tous les joueurs sont retournés au hall.',login_success:'Connexion réussie.',register_success:'Inscription réussie.',logged_out:'Déconnecté.',login_required:'Veuillez vous connecter d\\'abord.',stats_updated:'Statistiques mises à jour.',pvp_started:'Partie PVP commencée.',pvp_round_finished:'Tour PVP terminé.',pvp_match_finished:'Partie PVP terminée.',ranked_entry_ok:'Entrée classée acceptée.',ranked_entry_denied:'Entrée classée refusée.',room_join_failed:'Impossible de rejoindre la salle.',pvp_already_playing:'Cette partie PVP est déjà en cours.',pvp_seat_failed:'Impossible de créer les sièges PVP.',error:'Erreur',rules_classic_title:'Objectif : deux paysans contre un propriétaire',rules_classic_1:'Le mode classique a 3 joueurs. Un joueur devient le propriétaire et les deux autres sont paysans dans la même équipe.',rules_classic_2:'Le jeu a 54 cartes. Chaque joueur reçoit 17 cartes et 3 cartes restent face cachée pour le futur propriétaire.',rules_classic_3:'Le joueur qui reçoit la carte marquée commence la décision. Un joueur peut appeler propriétaire, puis les autres peuvent le voler.',rules_classic_4:'Après les enchères, le propriétaire prend les 3 cartes du bas dans sa main et joue en premier.',rules_classic_5:'Le propriétaire gagne en vidant sa main en premier. Les paysans gagnent si l\\'un d\\'eux vide sa main en premier.',rules_extended_title:'Mode trois contre un',rules_extended_1:'Le mode étendu a 4 joueurs. Un propriétaire combat trois paysans.',rules_extended_2:'Il utilise deux jeux. Chaque joueur reçoit 25 cartes et le propriétaire reçoit 8 cartes du bas.',rules_extended_3:'Le propriétaire peut révéler sa main. Révéler augmente le risque mais aussi le multiplicateur.',rules_extended_4:'Certaines mains peuvent être rapportées. Les rapports ajoutent des multiplicateurs supplémentaires.',rules_extended_5:'Les paysans ont des limites de bombes basées sur les enchères.',rules_extended_6:'Les suites sont plus flexibles dans ce mode, y compris les séquences A2345.',rules_turn_title:'Comment fonctionne un tour',rules_turn_1:'Le premier joueur d\\'un pli peut jouer n\\'importe quelle combinaison légale de sa main.',rules_turn_2:'Le joueur suivant doit jouer le même type de combinaison avec un rang supérieur, ou passer.',rules_turn_3:'Les bombes battent les combinaisons normales. Les fusées battent les bombes et toutes les combinaisons normales.',rules_turn_4:'Si tous les autres passent, le dernier joueur ayant joué des cartes commence un nouveau pli.',rules_turn_5:'Sélectionnez des cartes de votre main, puis appuyez sur Jouer. Si vous ne pouvez pas ou ne voulez pas battre la table, appuyez sur Passer.',rules_combo_title:'Combinaisons courantes',rules_combo_1:'Simple : une carte. Paire : deux cartes du même rang. Trio : trois cartes du même rang.',rules_combo_2:'Trio avec simple ou paire : trois cartes identiques plus une carte supplémentaire ou une paire.',rules_combo_3:'Suite : au moins cinq cartes simples consécutives. Suite de paires : au moins trois paires consécutives.',rules_combo_4:'Avion : trios consécutifs, avec optionnellement des simples ou paires supplémentaires.',rules_combo_5:'Bombe : quatre cartes ou plus du même rang. Fusée : jokers ensemble, la combinaison la plus forte.',rules_scoring_title:'Score',rules_scoring_1:'Les parties classées locales commencent à 1200 points et nécessitent au moins 200 points pour entrer.',rules_scoring_2:'Si votre score est inférieur à 200, le jeu peut vous recharger à 1200 points jusqu\\'à deux fois par jour.',rules_scoring_3:'Le score de base classé est 50. Le score de base PVP est 1.',rules_scoring_4:'L\\'enchère est le multiplicateur de base. Les bombes, fusées, révélations, redistributions, rapports, printemps, printemps inversé et marqueur joker peuvent encore doubler le score.',rules_scoring_5:'Les parties amicales enregistrent les victoires et défaites mais ne changent pas le score.',rules_pvp_title:'PVP en ligne',rules_pvp_1:'Le propriétaire crée une salle, définit optionnellement un mot de passe, choisit deux-contre-un ou trois-contre-un, et choisit le nombre de tours.',rules_pvp_2:'Les autres joueurs recherchent dans la liste des salles, entrent le mot de passe si nécessaire, et rejoignent la salle.',rules_pvp_3:'Quand les sièges sont pleins, le propriétaire commence la partie. Les joueurs jouent à tour de rôle via l\\'interface web.',rules_pvp_4:'Après chaque tour, les scores sont calculés automatiquement. Quand tous les tours sont terminés, le score total le plus élevé gagne.',rank:'Classement',room_name_length:'Le nom de la salle doit contenir entre 2 et 30 caractères.',invalid_mode:'Mode de jeu non valide.',invalid_rounds:'Les tours doivent être entre 1 et 50.',room_name_exists:'Ce nom de salle existe déjà. Veuillez en choisir un autre.',room_not_found:'Salle introuvable.',room_not_lobby:'La salle a déjà commencé ou est terminée, impossible de rejoindre.',wrong_password:'Mot de passe de salle incorrect.',already_in_room:'Vous êtes déjà dans cette salle.',room_full:'Salle pleine.',not_owner:'Seul le propriétaire de la salle peut effectuer cette action.',room_already_started:'La salle ne peut pas être redémarrée.',not_enough_players:'Pas assez de joueurs.',room_not_playing:'La salle n\\'est pas en cours de jeu.',landlord_not_in_room:'Le propriétaire n\\'est pas dans la salle.',match_completed:'Partie terminée.',round_finished:'Tour terminé.',player_disconnected:'{username} s\\'est déconnecté.',player_reconnected:'{username} s\\'est reconnecté.',all_disconnected:'Tous les joueurs se sont déconnectés. La salle sera nettoyée dans 5 minutes.',invalid_combo:'Combinaison de cartes non valide.',not_your_turn:'Ce n\\'est pas votre tour.',seat_taken:'Ce siège est déjà pris.',room_full_players:'La salle est pleine.',game_already_started:'La partie a déjà commencé.',cannot_pass_first:'Vous êtes le premier joueur de ce pli et ne pouvez pas passer.',cards_not_in_hand:'Certaines cartes sélectionnées ne sont pas dans votre main.',must_play_higher:'Vous devez jouer une combinaison plus élevée du même type.',landlord_not_selected:'Le propriétaire n\\'a pas encore été sélectionné.',bidding_in_progress:'Les enchères sont en cours.',not_in_room:'Vous n\\'êtes pas dans cette salle.'},
 pt:{app_title:'Dou Dizhu WebSocket',login:'Entrar',register:'Registrar',username:'Usuário',password:'Senha',password_hint:'Pressione Control+P neste campo para mostrar ou ocultar a senha.',ai_demo:'Partida IA',start_demo:'Iniciar demo',welcome:'Bem-vindo',local_ai:'Partida local com IA',stats:'Estatísticas',online_pvp:'PVP online',rules:'Regras',logout:'Sair',start_game:'Iniciar jogo',back:'Voltar',create_room:'Criar sala',join_room:'Entrar na sala',disband_room:'Dissolver sala',room_name:'Nome da sala',room_password_optional:'Senha da sala (opcional)',refresh:'Atualizar',rooms:'Salas',game_table:'Mesa de jogo',your_hand:'Sua mão',log:'Registro',mode_classic:'Dois contra um',mode_extended:'Três contra um',ranked:'Ranqueado',casual:'Casual',start_match:'Iniciar partida',seat:'Assento',empty:'Vazio',landlord:'Senhorio',farmer:'Fazendeiro',connected:'Conectado',disconnected:'Desconectado',waiting:'Aguardando',playing:'Jogando',finished:'Terminado',completed:'Completo',lobby:'Saguão',bid:'Apostar',no_bid:'Não apostar',call_landlord:'Ser senhorio',dont_call:'Não ser',rob_landlord:'Roubar senhorio',dont_rob:'Não roubar',reveal:'Revelar',dont_reveal:'Não revelar',report:'Relatar',dont_report:'Não relatar',play:'Jogar',pass:'Passar',round:'Rodada',score:'Pontuação',winner:'Vencedor',base_score:'Pontuação base',multiplier:'Multiplicador',bomb:'Bomba/Foguete',reveal_factor:'Revelar',redeal_factor:'Redistribuir',report_factor:'Relatório',marker_factor:'Curinga marcador',marker_card:'Carta marcada',marker_holder:'Portador da carta marcada',bottom_cards:'Cartas inferiores',spring:'Primavera',reverse_spring:'Primavera inversa',total:'Total',small_joker:'Curinga pequeno',big_joker:'Curinga grande',single:'Individual',pair:'Par',trio:'Trio',trio_single:'Trio com individual',trio_pair:'Trio com par',straight:'Sequência',pair_straight:'Sequência de pares',trio_straight:'Avião',airplane_single:'Avião com individuais',airplane_pair:'Avião com pares',four_two_single:'Quatro com dois',four_two_pair:'Quatro com dois pares',rocket:'Foguete',room_created:'Sala criada.',room_joined:'Você entrou na sala.',room_disbanded:'Sala dissolvida. Todos os jogadores voltaram ao saguão.',login_success:'Login bem-sucedido.',register_success:'Registro bem-sucedido.',logged_out:'Sessão encerrada.',login_required:'Por favor, faça login primeiro.',stats_updated:'Estatísticas atualizadas.',pvp_started:'Partida PVP iniciada.',pvp_round_finished:'Rodada PVP terminada.',pvp_match_finished:'Partida PVP terminada.',ranked_entry_ok:'Entrada ranqueada aceita.',ranked_entry_denied:'Entrada ranqueada negada.',room_join_failed:'Não foi possível entrar na sala.',pvp_already_playing:'Esta partida PVP já está em andamento.',pvp_seat_failed:'Não foi possível criar assentos PVP.',error:'Erro',rules_classic_title:'Objetivo: dois fazendeiros contra um senhorio',rules_classic_1:'O modo clássico tem 3 jogadores. Um jogador é o senhorio e os outros dois são fazendeiros no mesmo time.',rules_classic_2:'O baralho tem 54 cartas. Cada jogador recebe 17 cartas e 3 cartas ficam viradas para baixo para o futuro senhorio.',rules_classic_3:'O jogador que recebe a carta marcada inicia a decisão. Um jogador pode pedir senhorio, depois outros podem roubá-lo.',rules_classic_4:'Após a licitação, o senhorio pega as 3 cartas inferiores e joga primeiro.',rules_classic_5:'O senhorio vence esvaziando sua mão primeiro. Os fazendeiros vencem se qualquer um deles esvaziar sua mão primeiro.',rules_extended_title:'Modo três contra um',rules_extended_1:'O modo estendido tem 4 jogadores. Um senhorio luta contra três fazendeiros.',rules_extended_2:'Usa dois baralhos. Cada jogador recebe 25 cartas e o senhorio recebe 8 cartas inferiores.',rules_extended_3:'O senhorio pode revelar sua mão. Revelar aumenta o risco mas também o multiplicador.',rules_extended_4:'Algumas mãos podem ser relatadas. Relatos e relatos duplos adicionam multiplicadores extras.',rules_extended_5:'Os fazendeiros têm limites de bombas baseados na licitação.',rules_extended_6:'As sequências são mais flexíveis neste modo, incluindo sequências A2345.',rules_turn_title:'Como funciona um turno',rules_turn_1:'O primeiro jogador de uma rodada pode jogar qualquer combinação legal de sua mão.',rules_turn_2:'O próximo jogador deve jogar o mesmo tipo de combinação com uma classificação maior, ou passar.',rules_turn_3:'Bombas vencem combinações normais. Foguetes vencem bombas e todas as combinações normais.',rules_turn_4:'Se todos os outros passarem, o último jogador que jogou cartas inicia uma nova rodada.',rules_turn_5:'Selecione cartas da sua mão e pressione Jogar. Se não puder ou não quiser superar a mesa, pressione Passar.',rules_combo_title:'Combinações comuns',rules_combo_1:'Individual: uma carta. Par: duas cartas do mesmo valor. Trio: três cartas do mesmo valor.',rules_combo_2:'Trio com individual ou par: três iguais mais uma carta extra ou um par.',rules_combo_3:'Sequência: pelo menos cinco cartas individuais consecutivas. Sequência de pares: pelo menos três pares consecutivos.',rules_combo_4:'Avião: trios consecutivos, opcionalmente com individuais ou pares extras.',rules_combo_5:'Bomba: quatro ou mais cartas do mesmo valor. Foguete: curingas juntos, a combinação mais forte.',rules_scoring_title:'Pontuação',rules_scoring_1:'Partidas ranqueadas locais começam com 1200 pontos e exigem pelo menos 200 pontos para entrar.',rules_scoring_2:'Se sua pontuação estiver abaixo de 200, o jogo pode recarregar para 1200 pontos até duas vezes por dia.',rules_scoring_3:'A pontuação base ranqueada é 50. A pontuação base PVP é 1.',rules_scoring_4:'A aposta é o multiplicador base. Bombas, foguetes, revelação, redistribuições, relatórios, primavera, primavera inversa e curinga marcador podem dobrar a pontuação novamente.',rules_scoring_5:'Partidas casuais registram vitórias e derrotas mas não alteram a pontuação.',rules_pvp_title:'PVP online',rules_pvp_1:'O proprietário cria uma sala, opcionalmente define uma senha, escolhe dois-contra-um ou três-contra-um, e escolhe o número de rodadas.',rules_pvp_2:'Outros jogadores procuram na lista de salas, inserem a senha se necessário, e entram na sala.',rules_pvp_3:'Quando os assentos estão cheios, o proprietário inicia a partida. Os jogadores jogam em turnos através da interface web.',rules_pvp_4:'Após cada rodada, as pontuações são calculadas automaticamente. Quando todas as rodadas terminam, a maior pontuação total vence.',rank:'Classificação',room_name_length:'O nome da sala deve ter entre 2 e 30 caracteres.',invalid_mode:'Modo de jogo inválido.',invalid_rounds:'As rodadas devem ser entre 1 e 50.',room_name_exists:'Nome da sala já existe. Por favor, escolha outro.',room_not_found:'Sala não encontrada.',room_not_lobby:'A sala já começou ou terminou, não é possível entrar.',wrong_password:'Senha da sala incorreta.',already_in_room:'Você já está nesta sala.',room_full:'Sala cheia.',not_owner:'Apenas o proprietário da sala pode realizar esta ação.',room_already_started:'A sala não pode ser iniciada novamente.',not_enough_players:'Jogadores insuficientes.',room_not_playing:'A sala não está em jogo no momento.',landlord_not_in_room:'O nome de usuário do senhorio não está na sala.',match_completed:'Partida concluída.',round_finished:'Rodada concluída.',player_disconnected:'{username} desconectou.',player_reconnected:'{username} reconectou.',all_disconnected:'Todos os jogadores desconectaram. A sala será limpa em 5 minutos.',invalid_combo:'Combinação de cartas inválida.',not_your_turn:'Não é sua vez.',seat_taken:'Assento já ocupado.',room_full_players:'Sala cheia.',game_already_started:'O jogo já começou.',cannot_pass_first:'Você é o primeiro jogador desta rodada e não pode passar.',cards_not_in_hand:'Algumas cartas selecionadas não estão na sua mão.',must_play_higher:'Deve jogar uma combinação mais alta do mesmo tipo.',landlord_not_selected:'O senhorio ainda não foi selecionado.',bidding_in_progress:'Licitação em andamento.',not_in_room:'Você não está nesta sala.'},
 ru:{app_title:'Dou Dizhu WebSocket',login:'Вход',register:'Регистрация',username:'Имя пользователя',password:'Пароль',password_hint:'Нажмите Control+P в этом поле, чтобы показать или скрыть пароль.',ai_demo:'Игра с ИИ',start_demo:'Запустить демо',welcome:'Добро пожаловать',local_ai:'Локальная игра с ИИ',stats:'Статистика',online_pvp:'PVP онлайн',rules:'Правила',logout:'Выйти',start_game:'Начать игру',back:'Назад',create_room:'Создать комнату',join_room:'Войти в комнату',disband_room:'Расформировать',room_name:'Название комнаты',room_password_optional:'Пароль (необязательно)',refresh:'Обновить',rooms:'Комнаты',game_table:'Игровой стол',your_hand:'Ваша рука',log:'Журнал',mode_classic:'Двое против одного',mode_extended:'Трое против одного',ranked:'Рейтинговый',casual:'Обычный',start_match:'Начать матч',seat:'Место',empty:'Пусто',landlord:'Помещик',farmer:'Крестьянин',connected:'Подключен',disconnected:'Отключен',waiting:'Ожидание',playing:'Игра',finished:'Завершено',completed:'Выполнено',lobby:'Вестибюль',bid:'Ставка',no_bid:'Без ставки',call_landlord:'Стать помещиком',dont_call:'Не становиться',rob_landlord:'Отобрать помещика',dont_rob:'Не отбирать',reveal:'Раскрыть',dont_reveal:'Не раскрывать',report:'Доложить',dont_report:'Не докладывать',play:'Играть',pass:'Пас',round:'Раунд',score:'Счёт',winner:'Победитель',base_score:'Базовый счёт',multiplier:'Множитель',bomb:'Бомба/Ракета',reveal_factor:'Раскрытие',redeal_factor:'Пересдача',report_factor:'Доклад',marker_factor:'Маркер джокера',marker_card:'Отмеченная карта',marker_holder:'Держатель отмеченной карты',bottom_cards:'Нижние карты',spring:'Весна',reverse_spring:'Обратная весна',total:'Итого',small_joker:'Малый джокер',big_joker:'Большой джокер',single:'Одиночная',pair:'Пара',trio:'Тройка',trio_single:'Тройка с одиночной',trio_pair:'Тройка с парой',straight:'Стрит',pair_straight:'Парный стрит',trio_straight:'Самолёт',airplane_single:'Самолёт с одиночными',airplane_pair:'Самолёт с парами',four_two_single:'Четыре с двумя',four_two_pair:'Четыре с двумя парами',rocket:'Ракета',room_created:'Комната создана.',room_joined:'Вы вошли в комнату.',room_disbanded:'Комната расформирована. Все игроки возвращены в вестибюль.',login_success:'Вход выполнен.',register_success:'Регистрация выполнена.',logged_out:'Вы вышли.',login_required:'Пожалуйста, сначала войдите.',stats_updated:'Статистика обновлена.',pvp_started:'PVP матч начат.',pvp_round_finished:'PVP раунд завершён.',pvp_match_finished:'PVP матч завершён.',ranked_entry_ok:'Вход в рейтинговую игру разрешён.',ranked_entry_denied:'Вход в рейтинговую игру отклонён.',room_join_failed:'Не удалось войти в комнату.',pvp_already_playing:'Этот PVP матч уже идёт.',pvp_seat_failed:'Не удалось создать PVP места.',error:'Ошибка',rules_classic_title:'Цель: двое крестьян против одного помещика',rules_classic_1:'В классическом режиме 3 игрока. Один становится помещиком, двое других — крестьяне в одной команде.',rules_classic_2:'Колода из 54 карт. Каждый игрок получает по 17 карт, 3 карты остаются закрытыми для будущего помещика.',rules_classic_3:'Игрок с отмеченной картой начинает решение. Игрок может заявить помещика, затем другие могут его отобрать.',rules_classic_4:'После торгов помещик забирает 3 нижние карты в руку и ходит первым.',rules_classic_5:'Помещик выигрывает, первым опустошив руку. Крестьяне выигрывают, если любой из них первым опустошит руку.',rules_extended_title:'Режим трое против одного',rules_extended_1:'Расширенный режим имеет 4 игроков. Один помещик против трёх крестьян.',rules_extended_2:'Используются две колоды. Каждый игрок получает 25 карт, помещик получает 8 нижних карт.',rules_extended_3:'Помещик может раскрыть свою руку. Раскрытие увеличивает риск, но и множитель.',rules_extended_4:'Некоторые руки могут быть доложены. Доклад и двойной доклад добавляют дополнительные множители.',rules_extended_5:'У крестьян есть ограничения на бомбы в зависимости от ставок.',rules_extended_6:'Стриты более гибкие в этом режиме, включая последовательности A2345.',rules_turn_title:'Как проходит ход',rules_turn_1:'Первый игрок взятки может сыграть любую допустимую комбинацию из своей руки.',rules_turn_2:'Следующий игрок должен сыграть комбинацию того же типа с большим рангом или спасовать.',rules_turn_3:'Бомбы бьют обычные комбинации. Ракеты бьют бомбы и все обычные комбинации.',rules_turn_4:'Если все спасуют, последний игрок, сыгравший карты, начинает новую взятку.',rules_turn_5:'Выберите карты из руки и нажмите Играть. Если не можете или не хотите бить, нажмите Пас.',rules_combo_title:'Основные комбинации',rules_combo_1:'Одиночная: одна карта. Пара: две карты одного ранга. Тройка: три карты одного ранга.',rules_combo_2:'Тройка с одиночной или парой: три одинаковых плюс одна дополнительная карта или пара.',rules_combo_3:'Стрит: минимум пять последовательных одиночных карт. Парный стрит: минимум три последовательных пары.',rules_combo_4:'Самолёт: последовательные тройки, опционально с дополнительными одиночными или парами.',rules_combo_5:'Бомба: четыре или более карт одного ранга. Ракета: джокеры вместе, самая сильная комбинация.',rules_scoring_title:'Подсчёт очков',rules_scoring_1:'Локальные рейтинговые матчи начинаются с 1200 очков и требуют минимум 200 очков для входа.',rules_scoring_2:'Если ваш рейтинг ниже 200, игра может пополнить его до 1200 очков до двух раз в день.',rules_scoring_3:'Базовый рейтинговый счёт — 50. Базовый PVP счёт — 1.',rules_scoring_4:'Ставка — базовый множитель. Бомбы, ракеты, раскрытие, пересдачи, доклады, весна, обратная весна и маркер джокера могут снова удвоить счёт.',rules_scoring_5:'Обычные матчи записывают победы и поражения, но не меняют рейтинг.',rules_pvp_title:'PVP онлайн',rules_pvp_1:'Владелец создаёт комнату, опционально устанавливает пароль, выбирает режим и количество раундов.',rules_pvp_2:'Другие игроки ищут комнату в списке, вводят пароль при необходимости и присоединяются.',rules_pvp_3:'Когда все места заняты, владелец начинает матч. Игроки ходят по очереди через веб-интерфейс.',rules_pvp_4:'После каждого раунда очки рассчитываются автоматически. Когда все раунды завершены, побеждает наибольший общий счёт.',rank:'Рейтинг',room_name_length:'Название комнаты должно быть от 2 до 30 символов.',invalid_mode:'Недопустимый режим игры.',invalid_rounds:'Количество раундов должно быть от 1 до 50.',room_name_exists:'Название комнаты уже существует. Пожалуйста, выберите другое.',room_not_found:'Комната не найдена.',room_not_lobby:'Комната уже начала или закончила игру, нельзя присоединиться.',wrong_password:'Неверный пароль комнаты.',already_in_room:'Вы уже в этой комнате.',room_full:'Комната заполнена.',not_owner:'Только владелец комнаты может выполнить это действие.',room_already_started:'Комнату нельзя запустить повторно.',not_enough_players:'Недостаточно игроков.',room_not_playing:'В комнате сейчас не идёт игра.',landlord_not_in_room:'Имя помещика не в комнате.',match_completed:'Матч завершён.',round_finished:'Раунд завершён.',player_disconnected:'{username} отключился.',player_reconnected:'{username} переподключился.',all_disconnected:'Все игроки отключились. Комната будет очищена через 5 минут.',invalid_combo:'Недопустимая комбинация карт.',not_your_turn:'Не ваш ход.',seat_taken:'Место уже занято.',room_full_players:'Комната заполнена.',game_already_started:'Игра уже началась.',cannot_pass_first:'Вы первый игрок в этой взятке и не можете спасовать.',cards_not_in_hand:'Некоторые выбранные карты не в вашей руке.',must_play_higher:'Нужно сыграть более высокую комбинацию того же типа.',landlord_not_selected:'Помещик ещё не выбран.',bidding_in_progress:'Идут торги.',not_in_room:'Вы не в этой комнате.'},
 hi:{app_title:'Dou Dizhu WebSocket',login:'लॉगिन',register:'रजिस्टर',username:'उपयोगकर्ता नाम',password:'पासवर्ड',password_hint:'पासवर्ड दिखाने या छिपाने के लिए इस फ़ील्ड पर Control+P दबाएँ।',ai_demo:'AI मैच',start_demo:'डेमो शुरू करें',welcome:'स्वागतम्',local_ai:'स्थानीय AI मैच',stats:'आँकड़े',online_pvp:'ऑनलाइन PVP',rules:'नियम',logout:'लॉग आउट',start_game:'खेल शुरू करें',back:'वापस',create_room:'कक्ष बनाएँ',join_room:'कक्ष में शामिल हों',disband_room:'कक्ष भंग करें',room_name:'कक्ष का नाम',room_password_optional:'कक्ष पासवर्ड (वैकल्पिक)',refresh:'ताज़ा करें',rooms:'कक्ष',game_table:'खेल की मेज',your_hand:'आपके पत्ते',log:'लॉग',mode_classic:'दो बनाम एक',mode_extended:'तीन बनाम एक',ranked:'रैंक',casual:'सामान्य',start_match:'मैच शुरू करें',seat:'सीट',empty:'खाली',landlord:'ज़मींदार',farmer:'किसान',connected:'जुड़ा हुआ',disconnected:'डिस्कनेक्ट',waiting:'प्रतीक्षारत',playing:'खेल रहा है',finished:'समाप्त',completed:'पूर्ण',lobby:'लॉबी',bid:'बोली',no_bid:'बोली नहीं',call_landlord:'ज़मींदार बनें',dont_call:'न बनें',rob_landlord:'ज़मींदार छीनें',dont_rob:'न छीनें',reveal:'प्रकट करें',dont_reveal:'न प्रकट करें',report:'रिपोर्ट',dont_report:'रिपोर्ट नहीं',play:'चाल',pass:'पास',round:'राउंड',score:'अंक',winner:'विजेता',base_score:'आधार अंक',multiplier:'गुणक',bomb:'बम/रॉकेट',reveal_factor:'प्रकट',redeal_factor:'पुनर्वितरण',report_factor:'रिपोर्ट',marker_factor:'जोकर चिह्न',marker_card:'चिह्नित पत्ता',marker_holder:'चिह्नित पत्ता धारक',bottom_cards:'निचले पत्ते',spring:'स्प्रिंग',reverse_spring:'रिवर्स स्प्रिंग',total:'कुल',small_joker:'छोटा जोकर',big_joker:'बड़ा जोकर',single:'एकल',pair:'जोड़ी',trio:'तिकड़ी',trio_single:'एकल के साथ तिकड़ी',trio_pair:'जोड़ी के साथ तिकड़ी',straight:'सीधा',pair_straight:'जोड़ी सीधा',trio_straight:'प्लेन',airplane_single:'एकल के साथ प्लेन',airplane_pair:'जोड़ी के साथ प्लेन',four_two_single:'चार के साथ दो',four_two_pair:'चार के साथ दो जोड़ी',rocket:'रॉकेट',room_created:'कक्ष बनाया गया।',room_joined:'कक्ष में शामिल हुए।',room_disbanded:'कक्ष भंग किया गया। सभी खिलाड़ी लॉबी में लौट आए।',login_success:'लॉगिन सफल।',register_success:'पंजीकरण सफल।',logged_out:'लॉग आउट हो गए।',login_required:'कृपया पहले लॉगिन करें।',stats_updated:'आँकड़े अपडेट किए गए।',pvp_started:'PVP मैच शुरू हुआ।',pvp_round_finished:'PVP राउंड समाप्त।',pvp_match_finished:'PVP मैच समाप्त।',ranked_entry_ok:'रैंक प्रविष्टि स्वीकृत।',ranked_entry_denied:'रैंक प्रविष्टि अस्वीकृत।',room_join_failed:'कक्ष में शामिल नहीं हो सके।',pvp_already_playing:'यह PVP मैच पहले से चल रहा है।',pvp_seat_failed:'PVP सीट नहीं बन सकी।',error:'त्रुटि',rules_classic_title:'लक्ष्य: दो किसान बनाम एक ज़मींदार',rules_classic_1:'क्लासिक मोड में 3 खिलाड़ी हैं। एक ज़मींदार और दो किसान एक ही टीम में।',rules_classic_2:'54 पत्तों की गड्डी। प्रत्येक खिलाड़ी को 17 पत्ते मिलते हैं, 3 पत्ते भविष्य के ज़मींदार के लिए नीचे रखे जाते हैं।',rules_classic_3:'चिह्नित पत्ता पाने वाला खिलाड़ी ज़मींदार बनने का निर्णय शुरू करता है। बाद में अन्य खिलाड़ी ज़मींदार छीन सकते हैं।',rules_classic_4:'बोली के बाद, ज़मींदार 3 निचले पत्ते लेता है और पहले खेलता है।',rules_classic_5:'ज़मींदार पहले हाथ खाली करके जीतता है। किसान जीतते हैं यदि कोई भी किसान पहले हाथ खाली करे।',rules_extended_title:'तीन बनाम एक मोड',rules_extended_1:'विस्तारित मोड में 4 खिलाड़ी हैं। एक ज़मींदार तीन किसानों के खिलाफ।',rules_extended_2:'दो गड्डियाँ उपयोग होती हैं। प्रत्येक खिलाड़ी को 25 पत्ते, ज़मींदार को 8 निचले पत्ते।',rules_extended_3:'ज़मींदार अपने पत्ते प्रकट कर सकता है। प्रकट करने से जोखिम बढ़ता है लेकिन गुणक भी बढ़ता है।',rules_extended_4:'कुछ हाथ रिपोर्ट किए जा सकते हैं। रिपोर्ट और डबल रिपोर्ट अतिरिक्त गुणक जोड़ते हैं।',rules_extended_5:'किसानों की बम सीमाएँ बोली पर आधारित हैं।',rules_extended_6:'इस मोड में सीधे अधिक लचीले हैं, A2345 जैसे क्रम शामिल हैं।',rules_turn_title:'एक चाल कैसे काम करती है',rules_turn_1:'एक चाल का पहला खिलाड़ी अपने हाथ से कोई भी वैध संयोजन खेल सकता है।',rules_turn_2:'अगले खिलाड़ी को उसी प्रकार का संयोजन उच्च रैंक के साथ खेलना होगा, या पास करना होगा।',rules_turn_3:'बम सामान्य संयोजनों को हराते हैं। रॉकेट बम और सभी सामान्य संयोजनों को हराते हैं।',rules_turn_4:'यदि सभी पास करते हैं, तो अंतिम खिलाड़ी जिसने पत्ते खेले थे, नई चाल शुरू करता है।',rules_turn_5:'अपने हाथ से पत्ते चुनें, फिर चाल दबाएँ। यदि आप खेल नहीं सकते, तो पास दबाएँ।',rules_combo_title:'सामान्य संयोजन',rules_combo_1:'एकल: एक पत्ता। जोड़ी: समान रैंक के दो पत्ते। तिकड़ी: समान रैंक के तीन पत्ते।',rules_combo_2:'एकल या जोड़ी के साथ तिकड़ी: तीन समान पत्ते और एक अतिरिक्त पत्ता या जोड़ी।',rules_combo_3:'सीधा: कम से कम पाँच लगातार एकल पत्ते। जोड़ी सीधा: कम से कम तीन लगातार जोड़ियाँ।',rules_combo_4:'प्लेन: लगातार तिकड़ियाँ, वैकल्पिक अतिरिक्त एकल या जोड़ियों के साथ।',rules_combo_5:'बम: समान रैंक के चार या अधिक पत्ते। रॉकेट: जोकर एक साथ, सबसे मजबूत संयोजन।',rules_scoring_title:'अंक गणना',rules_scoring_1:'स्थानीय रैंक मैच 1200 अंकों से शुरू होते हैं और प्रविष्टि के लिए कम से कम 200 अंक आवश्यक हैं।',rules_scoring_2:'यदि आपका स्कोर 200 से कम है, तो खेल दिन में दो बार 1200 अंकों तक पुनः भर सकता है।',rules_scoring_3:'रैंक आधार अंक 50 है। PVP आधार अंक 1 है।',rules_scoring_4:'बोली आधार गुणक है। बम, रॉकेट, प्रकट, पुनर्वितरण, रिपोर्ट, स्प्रिंग, रिवर्स स्प्रिंग और जोकर चिह्न स्कोर को फिर से दोगुना कर सकते हैं।',rules_scoring_5:'सामान्य मैच जीत और हार दर्ज करते हैं लेकिन रैंक नहीं बदलते।',rules_pvp_title:'ऑनलाइन PVP',rules_pvp_1:'मालिक एक कक्ष बनाता है, वैकल्पिक पासवर्ड सेट करता है, मोड और राउंड की संख्या चुनता है।',rules_pvp_2:'अन्य खिलाड़ी कक्ष सूची खोजते हैं, आवश्यकतानुसार पासवर्ड डालते हैं और कक्ष में शामिल होते हैं।',rules_pvp_3:'जब सीटें भर जाती हैं, मालिक मैच शुरू करता है। खिलाड़ी वेब इंटरफ़ेस के माध्यम से बारी-बारी से खेलते हैं।',rules_pvp_4:'प्रत्येक राउंड के बाद अंक स्वचालित रूप से गणना होते हैं। सभी राउंड समाप्त होने पर, सबसे अधिक कुल अंक वाला जीतता है।',rank:'रैंक',room_name_length:'कक्ष का नाम 2 से 30 अक्षरों के बीच होना चाहिए।',invalid_mode:'अमान्य खेल मोड।',invalid_rounds:'राउंड 1 से 50 के बीच होने चाहिए।',room_name_exists:'कक्ष का नाम पहले से मौजूद है। कृपया दूसरा नाम चुनें।',room_not_found:'कक्ष नहीं मिला।',room_not_lobby:'कक्ष पहले ही शुरू या समाप्त हो चुका है, शामिल नहीं हो सकते।',wrong_password:'गलत कक्ष पासवर्ड।',already_in_room:'आप पहले से इस कक्ष में हैं।',room_full:'कक्ष भरा हुआ है।',not_owner:'केवल कक्ष मालिक ही यह कार्रवाई कर सकता है।',room_already_started:'कक्ष दोबारा शुरू नहीं किया जा सकता।',not_enough_players:'पर्याप्त खिलाड़ी नहीं हैं।',room_not_playing:'कक्ष में अभी खेल नहीं चल रहा है।',landlord_not_in_room:'ज़मींदार का नाम कक्ष में नहीं है।',match_completed:'मैच पूरा हुआ।',round_finished:'राउंड समाप्त।',player_disconnected:'{username} डिस्कनेक्ट हो गया।',player_reconnected:'{username} पुनः जुड़ा।',all_disconnected:'सभी खिलाड़ी डिस्कनेक्ट हो गए। कक्ष 5 मिनट में साफ कर दिया जाएगा।',invalid_combo:'अमान्य पत्ता संयोजन।',not_your_turn:'आपकी बारी नहीं है।',seat_taken:'सीट पहले से भरी है।',room_full_players:'कक्ष भरा हुआ है।',game_already_started:'खेल पहले ही शुरू हो चुका है।',cannot_pass_first:'आप इस चाल के पहले खिलाड़ी हैं और पास नहीं कर सकते।',cards_not_in_hand:'कुछ चयनित पत्ते आपके हाथ में नहीं हैं।',must_play_higher:'उसी प्रकार का उच्च संयोजन खेलना होगा।',landlord_not_selected:'ज़मींदार अभी चुना नहीं गया है।',bidding_in_progress:'बोली जारी है।',not_in_room:'आप इस कक्ष में नहीं हैं।'},
 bn:{app_title:'Dou Dizhu WebSocket',login:'লগইন',register:'নিবন্ধন',username:'ব্যবহারকারীর নাম',password:'পাসওয়ার্ড',password_hint:'পাসওয়ার্ড দেখাতে বা লুকাতে এই ঘরে Control+P চাপুন।',ai_demo:'AI ম্যাচ',start_demo:'ডেমো শুরু',welcome:'স্বাগতম',local_ai:'স্থানীয় AI ম্যাচ',stats:'পরিসংখ্যান',online_pvp:'অনলাইন PVP',rules:'নিয়ম',logout:'লগআউট',start_game:'খেলা শুরু',back:'ফিরুন',create_room:'কক্ষ তৈরি',join_room:'কক্ষে যোগ দিন',disband_room:'কক্ষ বাতিল',room_name:'কক্ষের নাম',room_password_optional:'কক্ষের পাসওয়ার্ড (ঐচ্ছিক)',refresh:'রিফ্রেশ',rooms:'কক্ষসমূহ',game_table:'খেলার টেবিল',your_hand:'আপনার তাস',log:'লগ',mode_classic:'দুই বনাম এক',mode_extended:'তিন বনাম এক',ranked:'র‌্যাঙ্ক',casual:'সাধারণ',start_match:'ম্যাচ শুরু',seat:'আসন',empty:'খালি',landlord:'জমিদার',farmer:'কৃষক',connected:'সংযুক্ত',disconnected:'বিচ্ছিন্ন',waiting:'অপেক্ষমাণ',playing:'খেলছে',finished:'শেষ',completed:'সম্পন্ন',lobby:'লবি',bid:'বিড',no_bid:'বিড নেই',call_landlord:'জমিদার হন',dont_call:'হবেন না',rob_landlord:'জমিদার ছিনিয়ে নিন',dont_rob:'ছিনবেন না',reveal:'প্রকাশ করুন',dont_reveal:'প্রকাশ করবেন না',report:'রিপোর্ট',dont_report:'রিপোর্ট নয়',play:'খেলুন',pass:'পাস',round:'রাউন্ড',score:'স্কোর',winner:'বিজয়ী',base_score:'মূল স্কোর',multiplier:'গুণক',bomb:'বোমা/রকেট',reveal_factor:'প্রকাশ',redeal_factor:'পুনর্বণ্টন',report_factor:'রিপোর্ট',marker_factor:'জোকার চিহ্ন',marker_card:'চিহ্নিত তাস',marker_holder:'চিহ্নিত তাসধারী',bottom_cards:'নিচের তাস',spring:'স্প্রিং',reverse_spring:'বিপরীত স্প্রিং',total:'মোট',small_joker:'ছোট জোকার',big_joker:'বড় জোকার',single:'একক',pair:'জোড়া',trio:'ত্রয়ী',trio_single:'একক সহ ত্রয়ী',trio_pair:'জোড়া সহ ত্রয়ী',straight:'স্ট্রেইট',pair_straight:'জোড়া স্ট্রেইট',trio_straight:'প্লেন',airplane_single:'একক সহ প্লেন',airplane_pair:'জোড়া সহ প্লেন',four_two_single:'চার সাথে দুই',four_two_pair:'চার সাথে দুই জোড়া',rocket:'রকেট',room_created:'কক্ষ তৈরি হয়েছে।',room_joined:'কক্ষে যোগ দিয়েছেন।',room_disbanded:'কক্ষ বাতিল হয়েছে। সব খেলোয়াড় লবিতে ফিরে এসেছেন।',login_success:'লগইন সফল।',register_success:'নিবন্ধন সফল।',logged_out:'লগআউট করেছেন।',login_required:'অনুগ্রহ করে প্রথমে লগইন করুন।',stats_updated:'পরিসংখ্যান আপডেট হয়েছে।',pvp_started:'PVP ম্যাচ শুরু হয়েছে।',pvp_round_finished:'PVP রাউন্ড শেষ।',pvp_match_finished:'PVP ম্যাচ শেষ।',ranked_entry_ok:'র‌্যাঙ্ক প্রবেশাধিকার গৃহীত।',ranked_entry_denied:'র‌্যাঙ্ক প্রবেশাধিকার প্রত্যাখ্যাত।',room_join_failed:'কক্ষে যোগ দেওয়া যায়নি।',pvp_already_playing:'এই PVP ম্যাচ ইতিমধ্যে চলছে।',pvp_seat_failed:'PVP আসন তৈরি করা যায়নি।',error:'ত্রুটি',rules_classic_title:'লক্ষ্য: দুই কৃষক বনাম এক জমিদার',rules_classic_1:'ক্লাসিক মোডে ৩ জন খেলোয়াড়। একজন জমিদার, বাকি দুইজন একই দলের কৃষক।',rules_classic_2:'৫৪ তাসের ডেক। প্রত্যেকে ১৭টি করে তাস পায়, ৩টি তাস ভবিষ্যত জমিদারের জন্য নিচে রাখা হয়।',rules_classic_3:'চিহ্নিত তাস পাওয়া খেলোয়াড় প্রথমে জমিদার হওয়ার সিদ্ধান্ত নেয়। পরে অন্যরা জমিদার ছিনিয়ে নিতে পারে।',rules_classic_4:'বিডিং শেষে, জমিদার ৩টি নিচের তাস নেয় এবং প্রথমে খেলে।',rules_classic_5:'জমিদার আগে হাত খালি করলে জেতে। কোনো কৃষক আগে হাত খালি করলে কৃষক দল জেতে।',rules_extended_title:'তিন বনাম এক মোড',rules_extended_1:'বর্ধিত মোডে ৪ জন খেলোয়াড়। একজন জমিদার তিন কৃষকের বিরুদ্ধে।',rules_extended_2:'দুই ডেক তাস ব্যবহার হয়। প্রত্যেকে ২৫টি তাস পায়, জমিদার ৮টি নিচের তাস পায়।',rules_extended_3:'জমিদার হাত প্রকাশ করতে পারে। প্রকাশ করলে ঝুঁকি বাড়ে কিন্তু গুণকও বাড়ে।',rules_extended_4:'কিছু হাত রিপোর্ট করা যায়। রিপোর্ট ও ডাবল রিপোর্ট অতিরিক্ত গুণক যোগ করে।',rules_extended_5:'কৃষকদের বিডিং অনুসারে বোমার সীমা আছে।',rules_extended_6:'এই মোডে স্ট্রেইট আরও নমনীয়, A2345 ক্রম সহ।',rules_turn_title:'এক চাল কীভাবে কাজ করে',rules_turn_1:'এক চালের প্রথম খেলোয়াড় হাত থেকে যেকোনো বৈধ সংযোজন খেলতে পারে।',rules_turn_2:'পরবর্তী খেলোয়াড়কে একই ধরনের সংযোজন উচ্চ র‌্যাঙ্কে খেলতে হবে, অথবা পাস করতে হবে।',rules_turn_3:'বোমা সাধারণ সংযোজনকে হারায়। রকেট বোমা ও সব সাধারণ সংযোজনকে হারায়।',rules_turn_4:'সবাই পাস করলে, শেষ যে খেলোয়াড় তাস খেলেছে সে নতুন চাল শুরু করে।',rules_turn_5:'হাত থেকে তাস বেছে খেলুন চাপুন। খেলতে না চাইলে পাস চাপুন।',rules_combo_title:'সাধারণ তাস সংযোজন',rules_combo_1:'একক: একটি তাস। জোড়া: সমান র‌্যাঙ্কের দুই তাস। ত্রয়ী: সমান র‌্যাঙ্কের তিন তাস।',rules_combo_2:'একক বা জোড়া সহ ত্রয়ী: তিন সমান তাস ও একটি অতিরিক্ত তাস বা জোড়া।',rules_combo_3:'স্ট্রেইট: অন্তত পাঁচটি ক্রমিক একক তাস। জোড়া স্ট্রেইট: অন্তত তিনটি ক্রমিক জোড়া।',rules_combo_4:'প্লেন: ক্রমিক ত্রয়ী, ঐচ্ছিক অতিরিক্ত একক বা জোড়া সহ।',rules_combo_5:'বোমা: সমান র‌্যাঙ্কের চার বা ততোধিক তাস। রকেট: জোকার একসাথে, সবচেয়ে শক্তিশালী সংযোজন।',rules_scoring_title:'স্কোরিং',rules_scoring_1:'স্থানীয় র‌্যাঙ্ক ম্যাচ ১২০০ পয়েন্ট থেকে শুরু হয় এবং প্রবেশের জন্য ন্যূনতম ২০০ পয়েন্ট প্রয়োজন।',rules_scoring_2:'স্কোর ২০০ এর নিচে হলে, খেলা দিনে দুইবার ১২০০ পয়েন্ট পর্যন্ত পুনরায় পূরণ করতে পারে।',rules_scoring_3:'র‌্যাঙ্ক মূল স্কোর ৫০। PVP মূল স্কোর ১।',rules_scoring_4:'বিড হল মূল গুণক। বোমা, রকেট, প্রকাশ, পুনর্বণ্টন, রিপোর্ট, স্প্রিং, বিপরীত স্প্রিং ও জোকার চিহ্ন স্কোর আবার দ্বিগুণ করতে পারে।',rules_scoring_5:'সাধারণ ম্যাচ জয়-পরাজয় নথিভুক্ত করে কিন্তু র‌্যাঙ্ক পরিবর্তন করে না।',rules_pvp_title:'অনলাইন PVP',rules_pvp_1:'মালিক একটি কক্ষ তৈরি করে, ঐচ্ছিক পাসওয়ার্ড সেট করে, মোড ও রাউন্ড সংখ্যা বেছে নেয়।',rules_pvp_2:'অন্য খেলোয়াড়রা কক্ষ তালিকা খোঁজে, প্রয়োজনমতো পাসওয়ার্ড দিয়ে কক্ষে যোগ দেয়।',rules_pvp_3:'আসন পূর্ণ হলে, মালিক ম্যাচ শুরু করে। খেলোয়াড়রা ওয়েব ইন্টারফেসের মাধ্যমে পালাক্রমে খেলে।',rules_pvp_4:'প্রতি রাউন্ডের পর স্কোর স্বয়ংক্রিয়ভাবে গণনা হয়। সব রাউন্ড শেষে, সর্বোচ্চ মোট স্কোর বিজয়ী হয়।',rank:'র‌্যাঙ্ক',room_name_length:'কক্ষের নাম ২ থেকে ৩০ অক্ষরের মধ্যে হতে হবে।',invalid_mode:'অবৈধ খেলা মোড।',invalid_rounds:'রাউন্ড ১ থেকে ৫০ এর মধ্যে হতে হবে।',room_name_exists:'কক্ষের নাম ইতিমধ্যে আছে। অনুগ্রহ করে অন্য নাম চয়ন করুন।',room_not_found:'কক্ষ পাওয়া যায়নি।',room_not_lobby:'কক্ষ ইতিমধ্যে শুরু বা শেষ হয়েছে, যোগ দেওয়া যাবে না।',wrong_password:'ভুল কক্ষ পাসওয়ার্ড।',already_in_room:'আপনি ইতিমধ্যে এই কক্ষে আছেন।',room_full:'কক্ষ পূর্ণ।',not_owner:'শুধুমাত্র কক্ষ মালিক এই কাজ করতে পারেন।',room_already_started:'কক্ষ আবার শুরু করা যাবে না।',not_enough_players:'পর্যাপ্ত খেলোয়াড় নেই।',room_not_playing:'কক্ষে এখন খেলা চলছে না।',landlord_not_in_room:'জমিদারের নাম কক্ষে নেই।',match_completed:'ম্যাচ সমাপ্ত।',round_finished:'রাউন্ড সমাপ্ত।',player_disconnected:'{username} বিচ্ছিন্ন হয়েছে।',player_reconnected:'{username} পুনরায় যুক্ত হয়েছে।',all_disconnected:'সব খেলোয়াড় বিচ্ছিন্ন। কক্ষ ৫ মিনিটে পরিষ্কার হবে।',invalid_combo:'অবৈধ তাস সংযোজন।',not_your_turn:'আপনার পালা নয়।',seat_taken:'আসন ইতিমধ্যে দখলকৃত।',room_full_players:'কক্ষ পূর্ণ।',game_already_started:'খেলা ইতিমধ্যে শুরু হয়েছে।',cannot_pass_first:'আপনি এই চালের প্রথম খেলোয়াড়, পাস করতে পারবেন না।',cards_not_in_hand:'কিছু নির্বাচিত তাস আপনার হাতে নেই।',must_play_higher:'একই ধরনের উচ্চতর সংযোজন খেলতে হবে।',landlord_not_selected:'জমিদার এখনো নির্বাচিত হয়নি।',bidding_in_progress:'বিডিং চলছে।',not_in_room:'আপনি এই কক্ষে নেই।'},
 ar:{app_title:'Dou Dizhu WebSocket',login:'تسجيل الدخول',register:'تسجيل حساب',username:'اسم المستخدم',password:'كلمة المرور',password_hint:'اضغط Control+P داخل هذا الحقل لإظهار كلمة المرور أو إخفائها.',ai_demo:'مباراة ذكاء اصطناعي',start_demo:'بدء العرض',welcome:'مرحباً',local_ai:'لعب محلي ضد الذكاء',stats:'الإحصاءات',online_pvp:'PVP عبر الإنترنت',rules:'القواعد',logout:'خروج',start_game:'بدء اللعبة',back:'رجوع',create_room:'إنشاء غرفة',join_room:'الانضمام لغرفة',disband_room:'حل الغرفة',room_name:'اسم الغرفة',room_password_optional:'كلمة مرور الغرفة (اختياري)',refresh:'تحديث',rooms:'الغرف',game_table:'طاولة اللعب',your_hand:'يدك',log:'سجل',mode_classic:'اثنان ضد واحد',mode_extended:'ثلاثة ضد واحد',ranked:'تنافسي',casual:'عادي',start_match:'بدء المباراة',seat:'مقعد',empty:'فارغ',landlord:'المالك',farmer:'فلاح',connected:'متصل',disconnected:'منفصل',waiting:'انتظار',playing:'يلعب',finished:'منتهي',completed:'مكتمل',lobby:'الردهة',bid:'مزايدة',no_bid:'بدون مزايدة',call_landlord:'أصبح المالك',dont_call:'لا تصبح',rob_landlord:'سرقة المالك',dont_rob:'لا تسرق',reveal:'كشف',dont_reveal:'لا تكشف',report:'إبلاغ',dont_report:'لا تبلغ',play:'لعب',pass:'تمرير',round:'جولة',score:'النتيجة',winner:'الفائز',base_score:'النتيجة الأساسية',multiplier:'مضاعف',bomb:'قنبلة/صاروخ',reveal_factor:'كشف',redeal_factor:'إعادة توزيع',report_factor:'إبلاغ',marker_factor:'علامة الجوكر',marker_card:'بطاقة معلّمة',marker_holder:'حامل البطاقة المعلّمة',bottom_cards:'البطاقات السفلية',spring:'ربيع',reverse_spring:'ربيع عكسي',total:'المجموع',small_joker:'جوكر صغير',big_joker:'جوكر كبير',single:'فردي',pair:'زوج',trio:'ثلاثي',trio_single:'ثلاثي مع فردي',trio_pair:'ثلاثي مع زوج',straight:'متتالية',pair_straight:'متتالية أزواج',trio_straight:'طائرة',airplane_single:'طائرة مع فرديات',airplane_pair:'طائرة مع أزواج',four_two_single:'أربعة مع اثنين',four_two_pair:'أربعة مع زوجين',rocket:'صاروخ',room_created:'تم إنشاء الغرفة.',room_joined:'انضممت إلى الغرفة.',room_disbanded:'تم حل الغرفة. عاد جميع اللاعبين إلى الردهة.',login_success:'تم تسجيل الدخول بنجاح.',register_success:'تم التسجيل بنجاح.',logged_out:'تم تسجيل الخروج.',login_required:'يرجى تسجيل الدخول أولاً.',stats_updated:'تم تحديث الإحصاءات.',pvp_started:'بدأت مباراة PVP.',pvp_round_finished:'انتهت جولة PVP.',pvp_match_finished:'انتهت مباراة PVP.',ranked_entry_ok:'تم قبول الدخول التنافسي.',ranked_entry_denied:'تم رفض الدخول التنافسي.',room_join_failed:'تعذر الانضمام إلى الغرفة.',pvp_already_playing:'هذه المباراة PVP قيد اللعب بالفعل.',pvp_seat_failed:'تعذر إنشاء مقاعد PVP.',error:'خطأ',rules_classic_title:'الهدف: فلاحان ضد مالك واحد',rules_classic_1:'الوضع الكلاسيكي يضم 3 لاعبين. واحد هو المالك والاثنان الآخران فلاحان في نفس الفريق.',rules_classic_2:'الطابق يحتوي على 54 بطاقة. كل لاعب يحصل على 17 بطاقة، و3 بطاقات تبقى مقلوبة للمالك المستقبلي.',rules_classic_3:'اللاعب الذي يحصل على البطاقة المعلّمة يبدأ القرار. يمكن للاعب أن يطلب المالك، ثم يمكن للآخرين سرقته.',rules_classic_4:'بعد المزايدة، يأخذ المالك 3 بطاقات سفلية في يده ويلعب أولاً.',rules_classic_5:'يفوز المالك بإفراغ يده أولاً. يفوز الفلاحون إذا أفرغ أي منهم يده أولاً.',rules_extended_title:'وضع ثلاثة ضد واحد',rules_extended_1:'الوضع الموسع يضم 4 لاعبين. مالك واحد ضد ثلاثة فلاحين.',rules_extended_2:'يستخدم طابقين من البطاقات. كل لاعب يحصل على 25 بطاقة، والمالك يحصل على 8 بطاقات سفلية.',rules_extended_3:'يمكن للمالك كشف يده. الكشف يزيد المخاطرة ولكن يزيد المضاعف أيضاً.',rules_extended_4:'بعض الأيدي يمكن الإبلاغ عنها. الإبلاغ والإبلاغ المزدوج يضيفان مضاعفات إضافية.',rules_extended_5:'للفلاحين حدود قنابل حسب المزايدة.',rules_extended_6:'المتتاليات أكثر مرونة في هذا الوضع، بما في ذلك تسلسلات A2345.',rules_turn_title:'كيف يعمل الدور',rules_turn_1:'أول لاعب في الجولة يمكنه لعب أي تركيبة قانونية من يده.',rules_turn_2:'اللاعب التالي يجب أن يلعب نفس نوع التركيبة برتبة أعلى، أو يمرر.',rules_turn_3:'القنابل تهزم التركيبات العادية. الصواريخ تهزم القنابل وجميع التركيبات العادية.',rules_turn_4:'إذا مرر الجميع، آخر لاعب لعب بطاقات يبدأ جولة جديدة.',rules_turn_5:'اختر بطاقات من يدك ثم اضغط لعب. إذا لم تستطع أو لا تريد التغلب على الطاولة، اضغط تمرير.',rules_combo_title:'التركيبات الشائعة',rules_combo_1:'فردي: بطاقة واحدة. زوج: بطاقتان من نفس الرتبة. ثلاثي: ثلاث بطاقات من نفس الرتبة.',rules_combo_2:'ثلاثي مع فردي أو زوج: ثلاث بطاقات متشابهة مع بطاقة إضافية أو زوج.',rules_combo_3:'متتالية: خمس بطاقات فردية متتالية على الأقل. متتالية أزواج: ثلاثة أزواج متتالية على الأقل.',rules_combo_4:'طائرة: ثلاثيات متتالية، اختيارياً مع فرديات أو أزواج إضافية.',rules_combo_5:'قنبلة: أربع بطاقات أو أكثر من نفس الرتبة. صاروخ: الجوكرات معاً، أقوى تركيبة.',rules_scoring_title:'النتيجة',rules_scoring_1:'المباريات التنافسية المحلية تبدأ من 1200 نقطة وتتطلب 200 نقطة على الأقل للدخول.',rules_scoring_2:'إذا كانت نتيجتك أقل من 200، يمكن للعبة إعادة تعبئتك إلى 1200 نقطة حتى مرتين في اليوم.',rules_scoring_3:'النتيجة الأساسية التنافسية هي 50. النتيجة الأساسية PVP هي 1.',rules_scoring_4:'المزايدة هي المضاعف الأساسي. القنابل، الصواريخ، الكشف، إعادة التوزيع، الإبلاغ، الربيع، الربيع العكسي وعلامة الجوكر يمكنها مضاعفة النتيجة مرة أخرى.',rules_scoring_5:'المباريات العادية تسجل الانتصارات والخسائر لكنها لا تغير التقييم.',rules_pvp_title:'PVP عبر الإنترنت',rules_pvp_1:'المالك ينشئ غرفة، يحدد كلمة مرور اختيارياً، يختار الوضع وعدد الجولات.',rules_pvp_2:'اللاعبون الآخرون يبحثون في قائمة الغرف، يدخلون كلمة المرور إذا لزم، وينضمون للغرفة.',rules_pvp_3:'عندما تمتلئ المقاعد، يبدأ المالك المباراة. يلعب اللاعبون بالدور عبر واجهة الويب.',rules_pvp_4:'بعد كل جولة، تُحسب النتائج تلقائياً. عندما تنتهي جميع الجولات، يفوز صاحب أعلى مجموع.',rank:'الترتيب',room_name_length:'يجب أن يكون اسم الغرفة بين 2 و 30 حرفاً.',invalid_mode:'وضع لعب غير صالح.',invalid_rounds:'يجب أن تكون الجولات بين 1 و 50.',room_name_exists:'اسم الغرفة موجود بالفعل. يرجى اختيار اسم آخر.',room_not_found:'الغرفة غير موجودة.',room_not_lobby:'الغرفة بدأت بالفعل أو انتهت، لا يمكن الانضمام.',wrong_password:'كلمة مرور الغرفة غير صحيحة.',already_in_room:'أنت بالفعل في هذه الغرفة.',room_full:'الغرفة ممتلئة.',not_owner:'فقط مالك الغرفة يمكنه القيام بهذا الإجراء.',room_already_started:'لا يمكن بدء الغرفة مرة أخرى.',not_enough_players:'عدد اللاعبين غير كافٍ.',room_not_playing:'الغرفة ليست قيد اللعب حالياً.',landlord_not_in_room:'اسم المالك غير موجود في الغرفة.',match_completed:'اكتملت المباراة.',round_finished:'انتهت الجولة.',player_disconnected:'{username} انفصل.',player_reconnected:'{username} أعاد الاتصال.',all_disconnected:'انفصل جميع اللاعبين. سيتم تنظيف الغرفة خلال 5 دقائق.',invalid_combo:'تركيبة بطاقات غير صالحة.',not_your_turn:'ليس دورك.',seat_taken:'المقعد محجوز بالفعل.',room_full_players:'الغرفة ممتلئة.',game_already_started:'اللعبة بدأت بالفعل.',cannot_pass_first:'أنت أول لاعب في هذه الجولة ولا يمكنك التمرير.',cards_not_in_hand:'بعض البطاقات المختارة ليست في يدك.',must_play_higher:'يجب لعب تركيبة أعلى من نفس النوع.',landlord_not_selected:'لم يتم اختيار المالك بعد.',bidding_in_progress:'المزايدة جارية.',not_in_room:'أنت لست في هذه الغرفة.'},
 ur:{app_title:'Dou Dizhu WebSocket',login:'لاگ اِن',register:'رجسٹر',username:'صارف کا نام',password:'پاس ورڈ',password_hint:'پاس ورڈ دکھانے یا چھپانے کے لیے اس خانے میں Control+P دبائیں۔',ai_demo:'AI میچ',start_demo:'ڈیمو شروع کریں',welcome:'خوش آمدید',local_ai:'مقامی AI میچ',stats:'اعدادوشمار',online_pvp:'آن لائن PVP',rules:'قواعد',logout:'لاگ آؤٹ',start_game:'کھیل شروع کریں',back:'واپس',create_room:'کمرہ بنائیں',join_room:'کمرے میں شامل ہوں',disband_room:'کمرہ ختم کریں',room_name:'کمرے کا نام',room_password_optional:'کمرے کا پاس ورڈ (اختیاری)',refresh:'تازہ کریں',rooms:'کمرے',game_table:'کھیل کی میز',your_hand:'آپ کے پتے',log:'لاگ',mode_classic:'دو بمقابلہ ایک',mode_extended:'تین بمقابلہ ایک',ranked:'درجہ بندی',casual:'عام',start_match:'مقابلہ شروع کریں',seat:'نشست',empty:'خالی',landlord:'زمیندار',farmer:'کسان',connected:'منسلک',disconnected:'منقطع',waiting:'انتظار میں',playing:'کھیل رہا ہے',finished:'ختم',completed:'مکمل',lobby:'لابی',bid:'بولی',no_bid:'بولی نہیں',call_landlord:'زمیندار بنیں',dont_call:'نہ بنیں',rob_landlord:'زمیندار چھینیں',dont_rob:'نہ چھینیں',reveal:'ظاہر کریں',dont_reveal:'ظاہر نہ کریں',report:'رپورٹ',dont_report:'رپورٹ نہیں',play:'چالیں',pass:'پاس',round:'راؤنڈ',score:'اسکور',winner:'فاتح',base_score:'بنیادی اسکور',multiplier:'ضارب',bomb:'بم/راکٹ',reveal_factor:'ظاہر',redeal_factor:'دوبارہ تقسیم',report_factor:'رپورٹ',marker_factor:'جوکر نشان',marker_card:'نشان زدہ پتہ',marker_holder:'نشان زدہ پتے والا',bottom_cards:'نچلے پتے',spring:'بہار',reverse_spring:'الٹی بہار',total:'کل',small_joker:'چھوٹا جوکر',big_joker:'بڑا جوکر',single:'اکیلا',pair:'جوڑا',trio:'تین',trio_single:'اکیلے کے ساتھ تین',trio_pair:'جوڑے کے ساتھ تین',straight:'سلسلہ',pair_straight:'جوڑوں کا سلسلہ',trio_straight:'ہوائی جہاز',airplane_single:'اکیلوں کے ساتھ ہوائی جہاز',airplane_pair:'جوڑوں کے ساتھ ہوائی جہاز',four_two_single:'چار کے ساتھ دو',four_two_pair:'چار کے ساتھ دو جوڑے',rocket:'راکٹ',room_created:'کمرہ بن گیا۔',room_joined:'آپ کمرے میں شامل ہو گئے۔',room_disbanded:'کمرہ ختم ہو گیا۔ تمام کھلاڑی لابی میں واپس آ گئے۔',login_success:'لاگ اِن کامیاب۔',register_success:'رجسٹریشن کامیاب۔',logged_out:'لاگ آؤٹ ہو گئے۔',login_required:'براہ کرم پہلے لاگ اِن کریں۔',stats_updated:'اعدادوشمار اپ ڈیٹ ہو گئے۔',pvp_started:'PVP مقابلہ شروع ہوا۔',pvp_round_finished:'PVP راؤنڈ ختم۔',pvp_match_finished:'PVP مقابلہ ختم۔',ranked_entry_ok:'درجہ بندی میں داخلہ قبول۔',ranked_entry_denied:'درجہ بندی میں داخلہ مسترد۔',room_join_failed:'کمرے میں شامل نہیں ہو سکے۔',pvp_already_playing:'یہ PVP مقابلہ پہلے سے جاری ہے۔',pvp_seat_failed:'PVP نشستیں نہیں بن سکیں۔',error:'خرابی',rules_classic_title:'مقصد: دو کسان بمقابلہ ایک زمیندار',rules_classic_1:'کلاسک موڈ میں 3 کھلاڑی ہیں۔ ایک زمیندار اور دو کسان ایک ہی ٹیم میں۔',rules_classic_2:'54 پتوں کی گڈی۔ ہر کھلاڑی کو 17 پتے ملتے ہیں، 3 پتے مستقبل کے زمیندار کے لیے نیچے رکھے جاتے ہیں۔',rules_classic_3:'نشان زدہ پتا پانے والا کھلاڑی زمیندار بننے کا فیصلہ شروع کرتا ہے۔ بعد میں دوسرے کھلاڑی زمیندار چھین سکتے ہیں۔',rules_classic_4:'بولی کے بعد، زمیندار 3 نچلے پتے لے کر پہلے کھیلتا ہے۔',rules_classic_5:'زمیندار پہلے ہاتھ خالی کر کے جیتتا ہے۔ کسان جیتتے ہیں اگر کوئی کسان پہلے ہاتھ خالی کرے۔',rules_extended_title:'تین بمقابلہ ایک موڈ',rules_extended_1:'وسیع موڈ میں 4 کھلاڑی ہیں۔ ایک زمیندار تین کسانوں کے خلاف۔',rules_extended_2:'دو گڈیاں استعمال ہوتی ہیں۔ ہر کھلاڑی کو 25 پتے، زمیندار کو 8 نچلے پتے۔',rules_extended_3:'زمیندار اپنے پتے ظاہر کر سکتا ہے۔ ظاہر کرنے سے خطرہ بڑھتا ہے لیکن ضارب بھی بڑھتا ہے۔',rules_extended_4:'کچھ ہاتھ رپورٹ کیے جا سکتے ہیں۔ رپورٹ اور ڈبل رپورٹ اضافی ضارب جوڑتے ہیں۔',rules_extended_5:'کسانوں کی بم کی حدود بولی پر مبنی ہیں۔',rules_extended_6:'اس موڈ میں سلسلے زیادہ لچکدار ہیں، A2345 ترتیب سمیت۔',rules_turn_title:'ایک باری کیسے کام کرتی ہے',rules_turn_1:'ایک باری کا پہلا کھلاڑی اپنے ہاتھ سے کوئی بھی جائز مجموعہ کھیل سکتا ہے۔',rules_turn_2:'اگلے کھلاڑی کو اسی قسم کا مجموعہ اعلیٰ رتبے کے ساتھ کھیلنا ہوگا، یا پاس کرنا ہوگا۔',rules_turn_3:'بم عام مجموعوں کو ہراتے ہیں۔ راکٹ بم اور تمام عام مجموعوں کو ہراتے ہیں۔',rules_turn_4:'اگر سب پاس کریں، تو آخری کھلاڑی جس نے پتے کھیلے تھے نئی باری شروع کرتا ہے۔',rules_turn_5:'اپنے ہاتھ سے پتے چنیں، پھر چالیں دبائیں۔ اگر آپ نہیں کھیل سکتے، تو پاس دبائیں۔',rules_combo_title:'عام پتوں کے مجموعے',rules_combo_1:'اکیلا: ایک پتا۔ جوڑا: ایک ہی رتبے کے دو پتے۔ تین: ایک ہی رتبے کے تین پتے۔',rules_combo_2:'اکیلے یا جوڑے کے ساتھ تین: تین ایک جیسے پتے اور ایک اضافی پتا یا جوڑا۔',rules_combo_3:'سلسلہ: کم از کم پانچ مسلسل اکیلے پتے۔ جوڑوں کا سلسلہ: کم از کم تین مسلسل جوڑے۔',rules_combo_4:'ہوائی جہاز: مسلسل تین، اختیاری اضافی اکیلوں یا جوڑوں کے ساتھ۔',rules_combo_5:'بم: ایک ہی رتبے کے چار یا زیادہ پتے۔ راکٹ: جوکر ایک ساتھ، سب سے مضبوط مجموعہ۔',rules_scoring_title:'اسکورنگ',rules_scoring_1:'مقامی درجہ بندی کے مقابلے 1200 پوائنٹس سے شروع ہوتے ہیں اور داخلے کے لیے کم از کم 200 پوائنٹس ضروری ہیں۔',rules_scoring_2:'اگر آپ کا اسکور 200 سے کم ہے، تو کھیل دن میں دو بار 1200 پوائنٹس تک دوبارہ بھر سکتا ہے۔',rules_scoring_3:'درجہ بندی کا بنیادی اسکور 50 ہے۔ PVP بنیادی اسکور 1 ہے۔',rules_scoring_4:'بولی بنیادی ضارب ہے۔ بم، راکٹ، ظاہر، دوبارہ تقسیم، رپورٹ، بہار، الٹی بہار اور جوکر نشان اسکور کو دوبارہ دوگنا کر سکتے ہیں۔',rules_scoring_5:'عام مقابلے جیت اور ہار درج کرتے ہیں لیکن درجہ بندی نہیں بدلتے۔',rules_pvp_title:'آن لائن PVP',rules_pvp_1:'مالک ایک کمرہ بناتا ہے، اختیاری پاس ورڈ سیٹ کرتا ہے، موڈ اور راؤنڈ کی تعداد چنتا ہے۔',rules_pvp_2:'دوسرے کھلاڑی کمروں کی فہرست تلاش کرتے ہیں، ضرورت پڑنے پر پاس ورڈ ڈال کر کمرے میں شامل ہوتے ہیں۔',rules_pvp_3:'جب نشستیں بھر جائیں، مالک مقابلہ شروع کرتا ہے۔ کھلاڑی ویب انٹرفیس کے ذریعے باری باری کھیلتے ہیں۔',rules_pvp_4:'ہر راؤنڈ کے بعد اسکور خودکار طور پر حساب ہوتے ہیں۔ جب تمام راؤنڈ ختم ہو جائیں، سب سے زیادہ کل اسکور والا جیتتا ہے۔',rank:'درجہ',room_name_length:'کمرے کا نام 2 سے 30 حروف کے درمیان ہونا چاہیے۔',invalid_mode:'غلط کھیل موڈ۔',invalid_rounds:'راؤنڈ 1 سے 50 کے درمیان ہونے چاہئیں۔',room_name_exists:'کمرے کا نام پہلے سے موجود ہے۔ براہ کرم دوسرا نام چنیں۔',room_not_found:'کمرہ نہیں ملا۔',room_not_lobby:'کمرہ پہلے ہی شروع یا ختم ہو چکا ہے، شامل نہیں ہو سکتے۔',wrong_password:'غلط کمرے کا پاس ورڈ۔',already_in_room:'آپ پہلے ہی اس کمرے میں ہیں۔',room_full:'کمرہ بھرا ہوا ہے۔',not_owner:'صرف کمرے کا مالک یہ کارروائی کر سکتا ہے۔',room_already_started:'کمرہ دوبارہ شروع نہیں کیا جا سکتا۔',not_enough_players:'کافی کھلاڑی نہیں ہیں۔',room_not_playing:'کمرے میں ابھی کھیل جاری نہیں ہے۔',landlord_not_in_room:'زمیندار کا نام کمرے میں نہیں ہے۔',match_completed:'مقابلہ مکمل۔',round_finished:'راؤنڈ ختم۔',player_disconnected:'{username} منقطع ہو گیا۔',player_reconnected:'{username} دوبارہ منسلک ہوا۔',all_disconnected:'تمام کھلاڑی منقطع ہو گئے۔ کمرہ 5 منٹ میں صاف کر دیا جائے گا۔',invalid_combo:'غلط پتوں کا مجموعہ۔',not_your_turn:'آپ کی باری نہیں ہے۔',seat_taken:'نشست پہلے سے لی گئی ہے۔',room_full_players:'کمرہ بھرا ہوا ہے۔',game_already_started:'کھیل پہلے ہی شروع ہو چکا ہے۔',cannot_pass_first:'آپ اس باری کے پہلے کھلاڑی ہیں اور پاس نہیں کر سکتے۔',cards_not_in_hand:'کچھ منتخب پتے آپ کے ہاتھ میں نہیں ہیں۔',must_play_higher:'اسی قسم کا اعلیٰ مجموعہ کھیلنا ہوگا۔',landlord_not_selected:'زمیندار ابھی منتخب نہیں ہوا۔',bidding_in_progress:'بولی جاری ہے۔',not_in_room:'آپ اس کمرے میں نہیں ہیں۔'},
};
const state={lang:'zh',username:'',ws:null,cards:[],selected:new Set(),seat:null,roomId:null,pvpRoom:null,matchType:''};
function tr(k,p={}){let s=(OV[state.lang]&&OV[state.lang][k])||BASE[k]||k;return s.replace(/\\{(\\w+)\\}/g,(_,x)=>p[x]??'')}
function localizeCard(c){const label=c.label||c;if(label==='小王')return tr('small_joker');if(label==='大王')return tr('big_joker');return label}
function localizeCombo(c){if(!c)return '';return tr(c.kind||c.display_name||'')+(c.sequence_length>1?' '+c.sequence_length:'')}
function send(d){if(state.ws&&state.ws.readyState===1){d.request_id='r'+Date.now()+Math.random();state.ws.send(JSON.stringify(d))}}
function logLine(k,p={}){const e=document.getElementById('log');e.textContent+='['+new Date().toLocaleTimeString()+'] '+tr(k,p)+'\\n';e.scrollTop=e.scrollHeight}
function logText(text){const e=document.getElementById('log');e.textContent+='['+new Date().toLocaleTimeString()+'] '+text+'\\n';e.scrollTop=e.scrollHeight}
function reason(p){return p.message || (p.message_key?tr(p.message_key,p.params||{}):tr('error'))}
function cardsText(cards){return (cards||[]).map(localizeCard).join(' ')}
function logHands(hands){(hands||[]).forEach(h=>logText(`${h.player_name} (${(h.cards||[]).length}): ${cardsText(h.cards)}`))}
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
function resetGameControls(){state.cards=[];state.selected.clear();renderHand();actionButtons('')}
function startAiDemo(){state.matchType='demo';resetGameControls();showView('game');send({type:'start_ai_demo',mode:val('demoMode'),rounds:Number(val('demoRounds')||1)})}
function startLocal(){state.matchType='local';resetGameControls();showView('game');send({type:'start_local_ai_match',mode:val('localMode'),match_type:val('localMatch')})}
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
case'pvp_room':state.matchType='pvp';state.pvpRoom=p.room&&p.room.room_name;showView('game');if(p.room){renderRooms([p.room]);var seats=(p.room.seats||[]).map(function(s){return{seat:s.seat,username:s.username,is_human:true,connected:true,hand_size:0,role:'farmer'}});renderPlayers(seats);document.getElementById('gameInfo').textContent=tr(p.room.mode==='extended'?'mode_extended':'mode_classic')+' \xb7 '+tr(p.room.status||'lobby');document.getElementById('startPvpBtn').classList.toggle('hidden',!(state.username===p.room.owner_username))}logText((p.ok?reason(p):`${tr('error')}: ${reason(p)}`));break;
case'pvp_room_disbanded':state.pvpRoom=null;state.roomId=null;state.seat=null;showView(state.username?'home':'auth');renderRooms([]);logText(reason(p));break;
case'pvp_match_started':state.pvpRoom=p.room&&p.room.room_name;showView('game');logText(p.ok?reason(p):`${tr('error')}: ${reason(p)}`);break;
case'local_game_created':state.matchType=p.match_type||state.matchType;state.roomId=p.room_id;state.seat=p.seat;showView('game');break;
case'room_created':case'room_joined':state.roomId=p.room_id;state.seat=p.seat;showView('game');logLine(m.type==='room_created'?'room_created':'room_joined');break;
case'room_state':renderPlayers(p.players);document.getElementById('gameInfo').textContent=`${tr(p.mode==='extended'?'mode_extended':'mode_classic')} · ${tr(p.state)}`;document.getElementById('startRoomBtn').classList.toggle('hidden',!(p.state==='waiting'&&p.host_username===state.username));break;
case'game_starting':showView('game');logText(`${tr('start_game')}: ${tr(p.mode==='extended'?'mode_extended':'mode_classic')}`);renderPlayers(p.players);break;
case'your_hand':case'your_cards':state.cards=p.cards||[];state.selected.clear();renderHand();break;
case'cards_dealt':logText(`${tr('marker_card')}: ${localizeCard({label:p.marked_card})} · ${tr('marker_holder')}: ${p.marker_holder_name||''} · ${tr('bottom_cards')}: ${p.bottom_count}`);logHands(p.player_hands);break;
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
case'landlord_assigned':logText(`${p.player_name}: ${tr('landlord')} · ${tr('bottom_cards')}: ${cardsText(p.bottom_cards)}`);if(p.hand&&p.hand.length)logHands([{player_name:p.player_name,cards:p.hand}]);break;
case'play_turn':if(state.matchType!=='demo')logText(`${tr('seat')} ${p.seat+1} ${p.player_name}: ${tr('play')}`);break;
case'new_round':logText(`${tr('round')}: ${p.leader_name}`);break;
case'no_bidder':logText(p.message||tr('no_bid'));break;
case'redeal':logText(p.message_key?tr(p.message_key,p.params||{}):(p.message||tr('redeal_factor')));break;
case'player_empty':logText(`${p.player_name}: ${tr('winner')}`);break;
case'play_action':logText(p.action==='play'?`${p.player_name} ${tr('play')}: ${comboText(p.combo,p.combo_display)} -> ${cardsText(p.cards_played)} | ${tr('your_hand')}: ${p.remaining_count}`:`${p.player_name}: ${tr('pass')} | ${tr('your_hand')}: ${p.remaining_count}`);break;
case'game_over':showSettlement(p);break;
case'pvp_round_result':if(p.room)renderRooms([p.room]);showSettlement(p);logLine(p.message_key||'pvp_round_finished');break;
case'stats_updated':logLine('stats_updated');break;
case'error':logText(`${tr('error')}: ${reason(p)}`);break;
}}
function showStats(s){if(!s)return;var aiRate=s.ai_win_rate!=null?(s.ai_win_rate*100).toFixed(1)+'%':'--';var pvpRate=s.pvp_win_rate!=null?(s.pvp_win_rate*100).toFixed(1)+'%':'--';var rank=s.rank!=null?'#'+s.rank:'--';logText(`${tr('score')}: ${s.rating||0} | AI ${tr('winner')}: ${aiRate} | PVP ${tr('winner')}: ${pvpRate} | ${tr('rank')}: ${rank}`)}
function showRules(p){showView('rules');document.getElementById('rulesContent').innerHTML=(p.sections||[]).map(s=>`<h3>${tr(s.title_key)}</h3><ul>${s.body_keys.map(k=>`<li>${tr(k)}</li>`).join('')}</ul>`).join('')}
function showSettlement(p){const s=p.settlement||{};actionButtons('');logLine('finished');document.getElementById('log').textContent+=`${tr('winner')}: ${p.winner_name||''}\
${tr('base_score')}: ${s.base_score||''} ${tr('multiplier')}: ${s.multiplier_factor||''} ${tr('total')}: ${s.total_score||''}\
${tr('bomb')}: ${s.bomb_multiplier||0} ${tr('reveal_factor')}: ${s.reveal_multiplier||0} ${tr('redeal_factor')}: ${s.redeal_multiplier||0} ${tr('report_factor')}: ${s.report_multiplier||0} ${tr('marker_factor')}: ${s.marker_multiplier||0} ${tr('spring')}: ${s.spring_multiplier||0} ${tr('reverse_spring')}: ${s.reverse_spring_multiplier||0}\
`}
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
                    "payload": {"message": "Invalid JSON", "message_key": "invalid_json"},
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
                    "message_key": message,
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
                    "message_key": message,
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
                    "message_key": message,
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
                    "message_key": message,
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
                        "payload": {"message": "Failed to join room", "message_key": "room_join_failed"},
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
                        "payload": {"message": "房间不存在", "message_key": "room_not_found"},
                    })
                    continue
                if room.state != "waiting":
                    await ws.send_json({
                        "type": "error",
                        "room_id": room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "游戏已开始", "message_key": "game_already_started"},
                    })
                    continue
                seat = room.add_player(username)
                if seat is None:
                    await ws.send_json({
                        "type": "error",
                        "room_id": room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "房间已满", "message_key": "room_full"},
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
                        "payload": {"message": "未在房间中", "message_key": "not_in_room", "message_key": "not_in_room"},
                    })
                    continue
                if room.host_username != (room.seats.get(current_seat, SeatInfo(username="")).username if current_seat is not None else ""):
                    await ws.send_json({
                        "type": "error",
                        "room_id": current_room_id,
                        "timestamp": "",
                        "request_id": request_id,
                        "payload": {"message": "只有房主可以开始", "message_key": "only_host_can_start"},
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
                        "payload": {"message": "至少需要一名真人玩家", "message_key": "need_human_player"},
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
                        "payload": {"message": "未在游戏中", "message_key": "not_in_game"},
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
                        "payload": {"message": "重连失败，token无效或已过期。", "message_key": "reconnect_failed"},
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
                        "payload": {"message": "未在房间中", "message_key": "not_in_room", "message_key": "not_in_room"},
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
                    "payload": {"message": f"未知操作: {msg_type}", "message_key": "unknown_action"},
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
    print(f"账号: {stats['username']} | 排名: #{stats['rank']} | 当前积分: {stats['rating']}")
    print(f"总战绩: {stats['wins']}胜 {stats['losses']}负，胜率 {stats['win_rate']:.1%}")
    print(f"AI对战胜率: {stats['ai_win_rate']:.1%} | PVP对战胜率: {stats['pvp_win_rate']:.1%}")
    print(f"积分赛: {stats['ranked_wins']}胜 {stats['ranked_losses']}负，胜率 {stats['ranked_win_rate']:.1%}")
    print(f"娱乐赛: {stats['casual_wins']}胜 {stats['casual_losses']}负，胜率 {stats['casual_win_rate']:.1%}")
    print(f"PVP: {stats['pvp_wins']}胜 {stats['pvp_losses']}负")
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
